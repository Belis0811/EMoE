import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from tqdm import tqdm

from eigen_moe import build, MoEConfig

class EarlyStopping:
    def __init__(self, patience=3, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0
        return self.early_stop

def accuracy(output, target, topk=(1,)):
    maxk = max(topk); B = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        res.append(correct[:k].reshape(-1).float().sum(0, keepdim=True).mul_(100.0/B).item())
    return res

def make_cifar100(root, image_size, batch_size, workers):
    normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    train_tf = T.Compose([T.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
                          T.RandomHorizontalFlip(), T.ToTensor(), normalize])
    test_tf  = T.Compose([T.Resize(int(image_size*256/224)),
                          T.CenterCrop(image_size), T.ToTensor(), normalize])
    train = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=train_tf)
    test  = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,  num_workers=workers, pin_memory=True)
    test_loader  = DataLoader(test,  batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, test_loader, 100

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="./data")
    ap.add_argument("--vit", type=str, default="vit_base_patch16_224")
    ap.add_argument("--pretrained", action="store_true", default=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--amp", action="store_true", default=True)


    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--r", type=int, default=128)
    ap.add_argument("--bottleneck", type=int, default=192)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--router-mode", choices=["soft", "top1", "top2"], default="top1")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--moe-blocks", type=str, default="last6")
    ap.add_argument("--apply-to-patches-only", action="store_true", default=True)
    ap.add_argument("--freeze-backbone", action="store_true", default=True)
    ap.add_argument("--unfreeze-layernorm", action="store_true", default=False)

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader, num_classes = make_cifar100(args.data_root, 224, args.batch_size, args.workers)

    cfg = MoEConfig(
        experts=args.experts,
        r=args.r,
        bottleneck=args.bottleneck,
        tau=args.tau,
        router_mode=args.router_mode,
        alpha=args.alpha,
        blocks=args.moe_blocks,
        apply_to_patches_only=args.apply_to_patches_only,
        freeze_backbone=args.freeze_backbone,
        unfreeze_layernorm=args.unfreeze_layernorm,
    )
    model = build(args.vit, num_classes=num_classes, pretrained=args.pretrained, cfg=cfg)
    
    try:
        checkpoint = torch.load(f'./checkpoints/eigen_moe_{args.vit}.pth')
        model.load_state_dict(checkpoint)
        print("Loaded checkpoint – continuing training …")
    except FileNotFoundError:
        print("No existing checkpoint found – training from scratch …")
    
    model.to(device)

    optim = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    early_stopping = EarlyStopping(patience=5, verbose=True)

    print(f"Backbone={args.vit} pretrained={args.pretrained} | experts={args.experts} r={args.r} mode={args.router_mode} blocks={args.moe_blocks}")
    print(f"Trainable params: {sum(p.numel() for p in model.trainable_parameters())/1e6:.3f} M")
    
    os.makedirs("logs", exist_ok=True)
    log_file = open(f"logs/training_log_{args.vit}.txt", "w")

    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, corr, loss_sum = 0, 0.0, 0.0
        for x, y in tqdm(train_loader, desc=f'Epoch {ep} Training'):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits, aux = model(x)
                loss = F.cross_entropy(logits, y) + aux
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            B = y.size(0)
            tot += B
            loss_sum += loss.item() * B
            corr += accuracy(logits, y, (1,))[0] * B / 100.0

        train_loss = loss_sum / tot
        train_acc = 100.0 * corr / tot

        # eval
        model.eval()
        tot, corr = 0, 0.0
        with torch.no_grad():
            for x, y in tqdm(test_loader, desc=f'Epoch {ep} Evaluating'):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits, _ = model(x)
                corr += accuracy(logits, y, (1,))[0] * y.size(0) / 100.0
                tot += y.size(0)
        val_acc = 100.0 * corr / tot
        best = max(best, val_acc)
        print(f"Epoch {ep:02d} | train_loss {train_loss:.4f} | train_acc {train_acc:.2f} | val_acc {val_acc:.2f} | best {best:.2f}")
        log_file.write(f"Epoch {ep:02d} | train_loss {train_loss:.4f} | train_acc {train_acc:.2f} | val_acc {val_acc:.2f} | best {best:.2f}\n")
        log_file.flush()
        
        if early_stopping(train_loss, model):
            print('Early stopping triggered')
            torch.save(model.state_dict(), f"checkpoints/eigen_moe_{args.vit}.pth")
            break
        
        os.makedirs("checkpoints", exist_ok=True)
        
        if ep % 5 == 0:
            torch.save(model.state_dict(), f"checkpoints/eigen_moe_{args.vit}.pth")
            print("Saved to checkpoints")
            
    log_file.close()

if __name__ == "__main__":
    main()