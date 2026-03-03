import torch
import torch.nn as nn
import torch.nn.functional as F


class ENNBasis(nn.Module):
    def __init__(self, d_in: int, d_out: int, r: int, ortho_lambda: float = 1e-3):
        super().__init__()
        assert r <= min(d_in, d_out)
        self.d_in, self.d_out, self.r = d_in, d_out, r
        self.ortho_lambda = ortho_lambda

        q = torch.empty(d_out, r)
        p = torch.empty(d_in, r)
        nn.init.orthogonal_(q)
        nn.init.orthogonal_(p)
        self.Q = nn.Parameter(q)
        self.P = nn.Parameter(p)
        self.log_lambda = nn.Parameter(torch.zeros(r))

    @torch.no_grad()
    def _qr_retract_(self):
        q_q, _ = torch.linalg.qr(self.Q, mode="reduced")
        q_p, _ = torch.linalg.qr(self.P, mode="reduced")
        self.Q.copy_(q_q)
        self.P.copy_(q_p)

    def ortho_penalty(self) -> torch.Tensor:
        ident = torch.eye(self.r, device=self.Q.device, dtype=self.Q.dtype)
        t1 = (self.Q.T @ self.Q - ident).pow(2).sum()
        t2 = (self.P.T @ self.P - ident).pow(2).sum()
        return self.ortho_lambda * (t1 + t2)

    def reconstruct_weight(self) -> torch.Tensor:
        lam = torch.diag_embed(self.log_lambda.exp())
        return self.Q @ lam @ self.P.T

    def project_out(self, h: torch.Tensor) -> torch.Tensor:
        return torch.einsum("dr,btd->btr", self.Q, h)


class AdapterExpert(nn.Module):
    def __init__(self, d_model, bottleneck=192):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, d_model, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        return self.up(self.act(self.down(x)))


class EigenRouter(nn.Module):
    def __init__(
        self,
        d_model: int,
        r: int,
        n_experts: int,
        tau: float = 1.0,
        topk: int = 0,
        ortho_lambda: float = 1e-3,
    ):
        super().__init__()
        self.n_experts, self.topk, self.tau = n_experts, topk, tau
        self.basis = ENNBasis(d_in=d_model, d_out=d_model, r=r, ortho_lambda=ortho_lambda)
        self.gamma = nn.Parameter(torch.ones(r))
        self.masks = nn.Parameter(torch.randn(n_experts, r))
        self.bias = nn.Parameter(torch.zeros(n_experts))

    def forward(self, h: torch.Tensor):
        if self.training:
            self.basis._qr_retract_()
        z = self.basis.project_out(h)
        e = z.pow(2)
        e = e / (e.sum(dim=-1, keepdim=True) + 1e-6)
        m = torch.softmax(self.masks, dim=0)
        logits = torch.einsum("btr,r,er->bte", e, self.gamma, m) + self.bias
        probs = F.softmax(logits / self.tau, dim=-1)
        ortho = self.basis.ortho_penalty()
        if self.topk and self.topk < self.n_experts:
            vals, idx = torch.topk(probs, k=self.topk, dim=-1)
            return probs, vals, idx, ortho
        return probs, None, None, ortho


class MoEAdapterBranch(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_experts: int = 8,
        r: int = 128,
        bottleneck: int = 192,
        tau: float = 1.0,
        router_mode: str = "soft",
        alpha: float = 1.0,
        apply_to_patches_only: bool = True,
        ortho_lambda: float = 1e-3,
    ):
        super().__init__()
        topk = 0 if router_mode == "soft" else (1 if router_mode == "top1" else 2)
        self.router = EigenRouter(d_model, r, n_experts, tau, topk, ortho_lambda)
        self.experts = nn.ModuleList([AdapterExpert(d_model, bottleneck) for _ in range(n_experts)])
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.apply_to_patches_only = apply_to_patches_only

    def forward(self, x: torch.Tensor):
        if self.apply_to_patches_only and x.dim() == 3 and x.size(1) >= 2:
            cls_tok, patches = x[:, :1, :], x[:, 1:, :]
            y, stats = self._forward_tokens(patches)
            return torch.cat([cls_tok, y], dim=1), stats
        return self._forward_tokens(x)

    def _forward_tokens(self, h: torch.Tensor):
        probs, vals, idx, ortho = self.router(h)
        stats = {
            "ortho_reg": ortho,
            "router_entropy": (-(probs * (probs.clamp_min(1e-9)).log())).sum(-1).mean(),
        }
        if idx is None:
            out = 0.0
            for e_id, expert in enumerate(self.experts):
                out = out + probs[..., e_id].unsqueeze(-1) * expert(h)
            return h + self.alpha * out, stats

        _, _, _ = h.shape
        k = idx.shape[-1]
        out = torch.zeros_like(h)
        with torch.no_grad():
            flat_idx = idx.reshape(-1, k)
            counts = torch.bincount(flat_idx.reshape(-1), minlength=len(self.experts))
            stats["assign_hist"] = counts.float() / counts.sum().clamp_min(1)

        for kk in range(k):
            ek = idx[..., kk]
            wk = vals[..., kk].unsqueeze(-1)
            for e_id, expert in enumerate(self.experts):
                mask = (ek == e_id).unsqueeze(-1)
                if mask.any():
                    out = out + mask * wk * expert(h)
        return h + self.alpha * out, stats
