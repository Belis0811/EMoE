from dataclasses import asdict
from pathlib import Path
import os
from typing import Dict, Optional

import torch
import torch.nn as nn

try:
    from huggingface_hub import HfApi, PyTorchModelHubMixin, hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
except ImportError:  # pragma: no cover - only used when huggingface_hub is unavailable
    HfApi = None  # type: ignore[assignment]
    PyTorchModelHubMixin = object  # type: ignore[assignment,misc]
    hf_hub_download = None  # type: ignore[assignment]
    EntryNotFoundError = FileNotFoundError  # type: ignore[assignment]

try:
    from .moe_model import MoEConfig, build
except ImportError:
    from moe_model import MoEConfig, build


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
            key = key[len("module.") :]
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
            weight_files = [
                name
                for name in repo_files
                if name.endswith((".pth", ".pt", ".bin", ".safetensors"))
            ]
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
