"""Backward-compatible facade for EMoE modules.

This file preserves the historical import path (`eigen_moe.py`) while the
implementation is split into smaller modules.
"""

try:
    from .hub_utils import (
        DEFAULT_HUB_CHECKPOINTS,
        HFEigenMoE,
        _clean_state_dict,
        default_hub_checkpoint_filename,
    )
    from .moe_model import EigenMoE, MoEConfig, build
except ImportError:
    from hub_utils import (
        DEFAULT_HUB_CHECKPOINTS,
        HFEigenMoE,
        _clean_state_dict,
        default_hub_checkpoint_filename,
    )
    from moe_model import EigenMoE, MoEConfig, build

__all__ = [
    "DEFAULT_HUB_CHECKPOINTS",
    "HFEigenMoE",
    "EigenMoE",
    "MoEConfig",
    "build",
    "default_hub_checkpoint_filename",
]
