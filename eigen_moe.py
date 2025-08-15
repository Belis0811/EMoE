from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple
import timm

class ENNBasis(nn.Module):
    def __init__(self, d_in: int, d_out: int, r: int, ortho_lambda: float = 1e-3):
        super().__init__()
        assert r <= min(d_in, d_out)
        self.d_in, self.d_out, self.r = d_in, d_out, r
        self.ortho_lambda = ortho_lambda

        Q = torch.empty(d_out, r)
        P = torch.empty(d_in,  r)
        nn.init.orthogonal_(Q)
        nn.init.orthogonal_(P)
        self.Q = nn.Parameter(Q)               
        self.P = nn.Parameter(P)                
        self.log_lambda = nn.Parameter(torch.zeros(r)) 

    @torch.no_grad()
    def _qr_retract_(self):
        qQ, _ = torch.linalg.qr(self.Q, mode='reduced')
        qP, _ = torch.linalg.qr(self.P, mode='reduced')
        self.Q.copy_(qQ); self.P.copy_(qP)

    def ortho_penalty(self) -> torch.Tensor:
        It = torch.eye(self.r, device=self.Q.device, dtype=self.Q.dtype)
        t1 = (self.Q.T @ self.Q - It).pow(2).sum()
        t2 = (self.P.T @ self.P - It).pow(2).sum()
        return self.ortho_lambda * (t1 + t2)

    def reconstruct_weight(self) -> torch.Tensor:
        lam = torch.diag_embed(self.log_lambda.exp())  
        return self.Q @ lam @ self.P.T         

    def project_out(self, h: torch.Tensor) -> torch.Tensor:
        return torch.einsum('dr,btd->btr', self.Q, h)  

class AdapterExpert(nn.Module):
    def __init__(self, d_model, bottleneck=192):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck, bias=False)
        self.up   = nn.Linear(bottleneck, d_model, bias=False)
        self.act  = nn.GELU()
    def forward(self, x): return self.up(self.act(self.down(x)))

class EigenRouter(nn.Module):
    def __init__(self, d_model: int, r: int, n_experts: int, tau: float = 1.0, topk: int = 0,
                 ortho_lambda: float = 1e-3):
        super().__init__()
        self.n_experts, self.topk, self.tau = n_experts, topk, tau
        self.basis = ENNBasis(d_in=d_model, d_out=d_model, r=r, ortho_lambda=ortho_lambda)
        self.gamma  = nn.Parameter(torch.ones(r))
        self.masks  = nn.Parameter(torch.randn(n_experts, r))  
        self.bias   = nn.Parameter(torch.zeros(n_experts))

    def forward(self, h: torch.Tensor):
        if self.training: self.basis._qr_retract_()
        z = self.basis.project_out(h)                       
        e = z.pow(2)
        e = e / (e.sum(dim=-1, keepdim=True) + 1e-6)     
        m = torch.softmax(self.masks, dim=0)               
        logits = torch.einsum('btr,r,er->bte', e, self.gamma, m) + self.bias
        probs  = F.softmax(logits / self.tau, dim=-1)    
        ortho  = self.basis.ortho_penalty()                
        if self.topk and self.topk < self.n_experts:
            vals, idx = torch.topk(probs, k=self.topk, dim=-1)  
            return probs, vals, idx, ortho
        return probs, None, None, ortho

class MoEAdapterBranch(nn.Module):
    def __init__(self, d_model: int, n_experts: int = 8, r: int = 128, bottleneck: int = 192,
                 tau: float = 1.0, router_mode: str = "soft", alpha: float = 1.0,
                 apply_to_patches_only: bool = True, ortho_lambda: float = 1e-3):
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
        else:
            return self._forward_tokens(x)

    def _forward_tokens(self, h: torch.Tensor):
        probs, vals, idx, ortho = self.router(h)
        stats = {"ortho_reg": ortho, "router_entropy": (-(probs * (probs.clamp_min(1e-9)).log())).sum(-1).mean()}
        if idx is None:
            out = 0.0
            for e_id, expert in enumerate(self.experts):
                out = out + probs[..., e_id].unsqueeze(-1) * expert(h)
            return h + self.alpha * out, stats
        B, T, D = h.shape; K = idx.shape[-1]
        out = torch.zeros_like(h)
        with torch.no_grad():
            flat_idx = idx.reshape(-1, K)
            counts = torch.bincount(flat_idx.reshape(-1), minlength=len(self.experts))
            stats["assign_hist"] = counts.float() / counts.sum().clamp_min(1)
        for k in range(K):
            ek = idx[..., k]               
            wk = vals[..., k].unsqueeze(-1) 
            for e_id, expert in enumerate(self.experts):
                mask = (ek == e_id).unsqueeze(-1)
                if mask.any(): out = out + mask * wk * expert(h)
        return h + self.alpha * out, stats


@dataclass
class MoEConfig:
    experts: int = 8
    r: int = 128
    bottleneck: int = 192
    tau: float = 1.0
    router_mode: str = "soft"  
    alpha: float = 1.0
    blocks: str = "last6"     
    apply_to_patches_only: bool = True
    ortho_lambda: float = 1e-3
    freeze_backbone: bool = True
    unfreeze_layernorm: bool = False

def _parse_block_indices(n_blocks: int, spec: str) -> List[int]:
    if spec == "all":   return list(range(n_blocks))
    if spec == "last6": return list(range(max(0, n_blocks - 6), n_blocks))
    if spec == "last4": return list(range(max(0, n_blocks - 4), n_blocks))
    return [i for i in map(int, spec.split(",")) if 0 <= i < n_blocks]

class EigenMoE(nn.Module):
    def __init__(self, vit: nn.Module, cfg: MoEConfig):
        super().__init__()
        self.vit, self.cfg = vit, cfg

        if cfg.freeze_backbone:
            for p in self.vit.parameters():
                p.requires_grad = False
        if cfg.unfreeze_layernorm:
            for m in self.vit.modules():
                if isinstance(m, nn.LayerNorm):
                    for p in m.parameters():
                        p.requires_grad = True

        d_model = getattr(self.vit, "embed_dim", None)
        if d_model is None:
            d_model = self.vit.blocks[0].norm1.normalized_shape[0]
        n_blocks = len(self.vit.blocks)
        self.block_ids = _parse_block_indices(n_blocks, cfg.blocks)

        self.branches = nn.ModuleDict()
        for i in self.block_ids:
            self.branches[str(i)] = MoEAdapterBranch(
                d_model=d_model,
                n_experts=cfg.experts,
                r=cfg.r,
                bottleneck=cfg.bottleneck,
                tau=cfg.tau,
                router_mode=cfg.router_mode,
                alpha=cfg.alpha,
                apply_to_patches_only=cfg.apply_to_patches_only,
                ortho_lambda=cfg.ortho_lambda,
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        vit = self.vit
        B = x.shape[0]
        x = vit.patch_embed(x)

        cls = vit.cls_token.expand(B, -1, -1)
        if getattr(vit, "dist_token", None) is not None:
            dist = vit.dist_token.expand(B, -1, -1)
            x = torch.cat([cls, dist, x], dim=1)
        else:
            x = torch.cat([cls, x], dim=1)

        if getattr(vit, "pos_embed", None) is not None:
            x = x + vit.pos_embed
        x = vit.pos_drop(x)

        aux_losses = []
        for i, blk in enumerate(vit.blocks):
            x = blk(x)
            key = str(i)
            if key in self.branches:
                x, stats = self.branches[key](x)
                aux_losses.append(stats["ortho_reg"])

        x = vit.norm(x)
        if hasattr(vit, "forward_head"):
            logits = vit.forward_head(x, pre_logits=False)
        else:
            logits = vit.head(x[:, 0])
        aux = torch.stack(aux_losses).sum() if aux_losses else logits.new_zeros(())
        return logits, aux

    def trainable_parameters(self):
        for p in self.parameters():
            if p.requires_grad: yield p

def build(
    vit: str = "vit_base_patch16_224",
    num_classes: int = 1000,
    pretrained: bool = True,
    cfg: Optional[MoEConfig] = None,
) -> EigenMoE:
    vit = timm.create_model(vit, pretrained=pretrained, num_classes=num_classes)
    if cfg is None:
        cfg = MoEConfig()
    return EigenMoE(vit, cfg)