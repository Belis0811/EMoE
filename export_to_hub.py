import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

try:
    from .eigen_moe import HFEigenMoE, default_hub_checkpoint_filename
except ImportError:
    from eigen_moe import HFEigenMoE, default_hub_checkpoint_filename


def parse_args():
    parser = argparse.ArgumentParser(description="Export and optionally push EMoE to Hugging Face Hub.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to local .pth checkpoint.")
    parser.add_argument("--output-dir", type=str, default="./hf_export", help="Directory for save_pretrained output.")
    parser.add_argument("--repo-id", type=str, default="", help="Hub repo id, e.g. anzheCheng/EMoE.")
    parser.add_argument("--push", action="store_true", help="Push exported files to --repo-id.")
    parser.add_argument("--upload-original-checkpoint", action="store_true", help="Also upload the raw .pth file.")
    parser.add_argument("--checkpoint-filename", type=str, default="", help="Filename to store raw checkpoint as in Hub.")
    parser.add_argument("--vit-model-name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--strict", action="store_true", help="Strict checkpoint loading.")

    # MoE hyperparameters
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--r", type=int, default=128)
    parser.add_argument("--bottleneck", type=int, default=192)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--router-mode", choices=["soft", "top1", "top2"], default="top1")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--blocks", type=str, default="last6")
    parser.add_argument("--apply-to-patches-only", dest="apply_to_patches_only", action="store_true")
    parser.add_argument("--no-apply-to-patches-only", dest="apply_to_patches_only", action="store_false")
    parser.add_argument("--ortho-lambda", type=float, default=1e-3)
    parser.add_argument("--freeze-backbone", dest="freeze_backbone", action="store_true")
    parser.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--unfreeze-layernorm", action="store_true", default=False)
    parser.add_argument("--backbone-pretrained", action="store_true", default=False)
    parser.set_defaults(apply_to_patches_only=True, freeze_backbone=True)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    moe_config = {
        "experts": args.experts,
        "r": args.r,
        "bottleneck": args.bottleneck,
        "tau": args.tau,
        "router_mode": args.router_mode,
        "alpha": args.alpha,
        "blocks": args.blocks,
        "apply_to_patches_only": args.apply_to_patches_only,
        "ortho_lambda": args.ortho_lambda,
        "freeze_backbone": args.freeze_backbone,
        "unfreeze_layernorm": args.unfreeze_layernorm,
    }

    model = HFEigenMoE(
        vit_model_name=args.vit_model_name,
        num_classes=args.num_classes,
        backbone_pretrained=args.backbone_pretrained,
        moe_config=moe_config,
    )
    missing, unexpected = model.load_checkpoint(
        str(checkpoint_path),
        map_location="cpu",
        strict=args.strict,
    )
    print(f"Loaded checkpoint: missing_keys={len(missing)} unexpected_keys={len(unexpected)}")

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    print(f"Saved Hub format model to: {args.output_dir}")

    if not args.push:
        return

    if not args.repo_id:
        raise ValueError("--repo-id is required when using --push.")

    print(f"Pushing to Hub repo: {args.repo_id}")
    model.push_to_hub(args.repo_id)

    if args.upload_original_checkpoint:
        upload_name = args.checkpoint_filename or default_hub_checkpoint_filename(args.vit_model_name)
        if not upload_name:
            upload_name = checkpoint_path.name
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(checkpoint_path),
            path_in_repo=upload_name,
            repo_id=args.repo_id,
            repo_type="model",
        )
        print(f"Uploaded original checkpoint as: {upload_name}")


if __name__ == "__main__":
    main()
