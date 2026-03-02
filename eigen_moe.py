from dataclasses import asdict, dataclass
from pathlib import Path
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple
import timm

try:
    from huggingface_hub import HfApi, PyTorchModelHubMixin, hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
except ImportError:  # pragma: no cover - only used when huggingface_hub is unavailable
    HfApi = None  # type: ignore[assignment]
    PyTorchModelHubMixin = object  # type: ignore[assignment,misc]
    hf_hub_download = None  # type: ignore[assignment]
    EntryNotFoundError = FileNotFoundError  # type: ignore[assignment]

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


DEFAULT_HUB_CHECKPOINTS = {
    "vit_base_patch16_224": "eigen_moe_vit_base_patch16_224_imagenet1k.pth",
    "vit_large_patch16_224.augreg_in21k_ft_in1k": "eigen_moe_vit_large_patch16_224.augreg_in21k_ft_in1k_imagenet1k.pth",
    "vit_huge_patch14_224_in21k": "eigen_moe_vit_huge_patch14_224_in21k_imagenet1k.pth",
}


def default_hub_checkpoint_filename(vit_model_name: str) -> Optional[str]:
    return DEFAULT_HUB_CHECKPOINTS.get(vit_model_name)


def _clean_state_dict(raw_checkpoint: Dict) -> Dict[str, torch.Tensor]:
    if not isinstance(raw_checkpoint, dict):
        raise TypeError(f"Expected checkpoint to be a dict, got {type(raw_checkpoint)}")

    for key in ("state_dict", "model_state_dict", "model"):
        if key in raw_checkpoint and isinstance(raw_checkpoint[key], dict):
            raw_checkpoint = raw_checkpoint[key]
            break

    cleaned = {}
    for key, value in raw_checkpoint.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            continue
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value
    if not cleaned:
        raise ValueError("No tensor weights were found in checkpoint.")
    return cleaned


class HFEigenMoE(nn.Module, PyTorchModelHubMixin):
    """Hugging Face Hub wrapper for EigenMoE checkpoints."""

    def __init__(
        self,
        vit_model_name: str = "vit_base_patch16_224",
        num_classes: int = 1000,
        backbone_pretrained: bool = False,
        moe_config: Optional[Dict] = None,
    ):
        super().__init__()
        cfg = MoEConfig(**(moe_config or {}))
        self.vit_model_name = vit_model_name
        self.num_classes = num_classes
        self.backbone_pretrained = backbone_pretrained
        self.moe_config = asdict(cfg)
        self.model = build(
            vit=vit_model_name,
            num_classes=num_classes,
            pretrained=backbone_pretrained,
            cfg=cfg,
        )

    def forward(self, pixel_values: torch.Tensor, return_aux: bool = False):
        logits, aux = self.model(pixel_values)
        if return_aux:
            return logits, aux
        return logits

    def load_checkpoint(
        self,
        checkpoint_path: str,
        map_location: str = "cpu",
        strict: bool = True,
    ):
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        state_dict = _clean_state_dict(checkpoint)
        return self._load_state_dict_flexible(state_dict, strict=strict)

    def _load_state_dict_flexible(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):
        try:
            return self.load_state_dict(state_dict, strict=strict)
        except RuntimeError as wrapper_err:
            try:
                return self.model.load_state_dict(state_dict, strict=strict)
            except RuntimeError as inner_err:
                raise RuntimeError(
                    "Failed to load checkpoint into both wrapper and inner EigenMoE model.\n"
                    f"Wrapper error: {wrapper_err}\n"
                    f"Inner model error: {inner_err}"
                ) from inner_err

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[str],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: Optional[bool],
        local_files_only: bool,
        token: Optional[str],
        map_location: str = "cpu",
        strict: bool = False,
        **model_kwargs,
    ):
        checkpoint_filename = model_kwargs.pop("checkpoint_filename", None)
        model = cls(**model_kwargs)

        checkpoint_path = cls._resolve_checkpoint_path(
            model_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
            checkpoint_filename=checkpoint_filename,
            vit_model_name=model.vit_model_name,
        )

        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(checkpoint_path, device=map_location)
        else:
            raw = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
            state_dict = _clean_state_dict(raw)

        model._load_state_dict_flexible(state_dict, strict=strict)
        return model

    @classmethod
    def _resolve_checkpoint_path(
        cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[str],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: Optional[bool],
        local_files_only: bool,
        token: Optional[str],
        checkpoint_filename: Optional[str],
        vit_model_name: str,
    ) -> str:
        if os.path.isdir(model_id):
            return cls._resolve_local_checkpoint(model_id, checkpoint_filename, vit_model_name)
        return cls._resolve_remote_checkpoint(
            model_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
            checkpoint_filename=checkpoint_filename,
            vit_model_name=vit_model_name,
        )

    @staticmethod
    def _resolve_local_checkpoint(
        model_dir: str,
        checkpoint_filename: Optional[str],
        vit_model_name: str,
    ) -> str:
        base = Path(model_dir)
        if checkpoint_filename:
            candidates = [checkpoint_filename]
        else:
            candidates = ["model.safetensors", "pytorch_model.bin"]
            default_name = default_hub_checkpoint_filename(vit_model_name)
            if default_name:
                candidates.append(default_name)

        for filename in candidates:
            path = base / filename
            if path.exists():
                return str(path)

        pth_files = sorted(base.glob("*.pth"))
        if pth_files:
            return str(pth_files[0])

        raise FileNotFoundError(
            f"Could not find a checkpoint in local directory: {model_dir}. "
            f"Tried {candidates} and '*.pth'."
        )

    @staticmethod
    def _resolve_remote_checkpoint(
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[str],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: Optional[bool],
        local_files_only: bool,
        token: Optional[str],
        checkpoint_filename: Optional[str],
        vit_model_name: str,
    ) -> str:
        if hf_hub_download is None:
            raise ImportError("huggingface_hub is required to download checkpoints from the Hub.")

        if checkpoint_filename:
            candidates = [checkpoint_filename]
        else:
            candidates = ["model.safetensors", "pytorch_model.bin"]
            default_name = default_hub_checkpoint_filename(vit_model_name)
            if default_name:
                candidates.append(default_name)

        seen = set()
        unique_candidates = []
        for name in candidates:
            if name not in seen:
                seen.add(name)
                unique_candidates.append(name)

        for filename in unique_candidates:
            try:
                return hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except EntryNotFoundError:
                continue

        if HfApi is not None:
            api = HfApi(token=token)
            repo_files = api.list_repo_files(repo_id=model_id, revision=revision)
            weight_files = [name for name in repo_files if name.endswith((".pth", ".pt", ".bin", ".safetensors"))]
            if weight_files:
                return hf_hub_download(
                    repo_id=model_id,
                    filename=weight_files[0],
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )

        raise FileNotFoundError(
            f"No compatible checkpoint found in Hub repo '{model_id}'. "
            f"Tried {unique_candidates} and a fallback scan for *.pth/*.pt/*.bin/*.safetensors."
        )
