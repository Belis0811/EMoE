# EMoE

Eigenbasis-Guided Mixture-of-Experts (EMoE) with Hugging Face Hub support.

## Install

```bash
pip install -U torch timm huggingface_hub safetensors
```

## Load from Hugging Face Hub

```python
import torch
from eigen_moe import HFEigenMoE

model = HFEigenMoE.from_pretrained(
    "anzheCheng/EMoE",
    vit_model_name="vit_base_patch16_224",
    num_classes=1000,
    checkpoint_filename="eigen_moe_vit_base_patch16_224_imagenet1k.pth",  # optional for this repo
    strict=False,
)
model.eval()

pixel_values = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    logits = model(pixel_values)
print(logits.shape)
```

`checkpoint_filename` is optional when the ViT name is one of:
- `vit_base_patch16_224`
- `vit_large_patch16_224.augreg_in21k_ft_in1k`
- `vit_huge_patch14_224_in21k`

## Export and Push

Convert a local `.pth` to standard Hub files (`model.safetensors`, `config.json`) and optionally push:

```bash
python export_to_hub.py \
  --checkpoint ./checkpoints/eigen_moe_vit_base_patch16_224.pth \
  --output-dir ./hf_export \
  --vit-model-name vit_base_patch16_224 \
  --num-classes 1000
```

Push directly:

```bash
python export_to_hub.py \
  --checkpoint ./checkpoints/eigen_moe_vit_base_patch16_224.pth \
  --repo-id anzheCheng/EMoE \
  --push \
  --upload-original-checkpoint \
  --vit-model-name vit_base_patch16_224 \
  --num-classes 1000
```
