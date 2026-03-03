from dataclasses import dataclass
from typing import List, Optional, Tuple

import timm
import torch
import torch.nn as nn

try:
    from .moe_layers import MoEAdapterBranch
except ImportError:
    from moe_layers import MoEAdapterBranch


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
    if spec == "all":
        return list(range(n_blocks))
    if spec == "last6":
        return list(range(max(0, n_blocks - 6), n_blocks))
    if spec == "last4":
        return list(range(max(0, n_blocks - 4), n_blocks))
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
        bsz = x.shape[0]
        x = vit.patch_embed(x)

        cls = vit.cls_token.expand(bsz, -1, -1)
        if getattr(vit, "dist_token", None) is not None:
            dist = vit.dist_token.expand(bsz, -1, -1)
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
            if p.requires_grad:
                yield p


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
