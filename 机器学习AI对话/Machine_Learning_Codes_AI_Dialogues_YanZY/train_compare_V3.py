#!/usr/bin/env python3
"""
train_mnist_improved.py

Improved MNIST training with:
 - deeper CNN backbone (32->64->128), BN, ReLU
 - AdaptiveAvgPool (GAP) + small classifier
 - data augmentation (rotation, affine translate, random erasing)
 - optimizer: SGD+OneCycleLR (default) or AdamW
 - mixed precision (AMP)
 - validation split, best-model checkpointing
 - torchsummary print
"""
import argparse
import time
import random
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchsummary import summary

# -------------------------
# utils
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # If you want deterministic behavior (may reduce performance):
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def params_size_mb(num_params: int):
    return num_params * 4 / (1024 ** 2)

# -------------------------
# model
# -------------------------
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=10, dropout_p=0.5):
        super().__init__()
        # conv blocks: conv -> bn -> relu [-> conv -> bn -> relu] -> pool
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 28 -> 14
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 14 -> 7
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # optional extra conv to increase capacity
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # no further pooling to keep feature map >= 7x7 (we already at 7x7)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # -> (N,128,1,1)
        self.classifier = nn.Sequential(
            nn.Flatten(),               # -> (N,128)
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

# -------------------------
# training / evaluation
# -------------------------
def train_one_epoch(model, device, loader, criterion, optimizer, scaler, use_amp=True):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    t0 = time.perf_counter()
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    epoch_time = time.perf_counter() - t0
    return running_loss / total, correct / total, epoch_time

@torch.no_grad()
def evaluate(model, device, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

# -------------------------
# main run
# -------------------------
def run(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Device: {device} | opt: {args.opt} | batch: {args.batch} | seed: {args.seed}")

    # data transforms: stronger augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomRotation(12),  # degrees
        transforms.RandomAffine(0, translate=(0.03, 0.03)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # datasets
    dataset_full = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=test_transform)

    # split train -> train/val
    val_size = int(len(dataset_full) * args.val_ratio)
    train_size = len(dataset_full) - val_size
    train_dataset, val_dataset = random_split(dataset_full, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(args.seed))
    # Note: val_dataset inherits train_transform; replace transform for val
    val_dataset.dataset.transform = test_transform

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # model
    model = ImprovedCNN(num_classes=10, dropout_p=args.dropout)
    model = model.to(device)

    # summary
    try:
        summary(model, (1, 28, 28), device=str(device))
    except Exception:
        pass
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,} | Params size (MB, float32): {params_size_mb(total_params):.2f}")

    # criterion & optimizer
    criterion = nn.CrossEntropyLoss()
    if args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.max_lr, momentum=0.9, weight_decay=args.wd)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr,
                                                  steps_per_epoch=len(train_loader), epochs=args.epochs,
                                                  pct_start=0.1, anneal_strategy='cos')
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.wd)
        # For AdamW we can still use OneCycleLR; keep pct_start small
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr,
                                                  steps_per_epoch=len(train_loader), epochs=args.epochs,
                                                  pct_start=0.1, anneal_strategy='cos')

    # amp scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda' and not args.no_amp_disable))

    best_val_acc = 0.0
    best_state = None
    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[], 'epoch_time':[]}

    # warm-up: optional small run to stabilize cuDNN selection
    torch.backends.cudnn.benchmark = True

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, epoch_time = train_one_epoch(model, device, train_loader,
                                                            criterion, optimizer, scaler, use_amp=(device.type=='cuda' and not args.no_amp_disable))
        val_loss, val_acc = evaluate(model, device, val_loader, criterion)

        if scheduler is not None:
            scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_time'].append(epoch_time)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            # save checkpoint
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save({'epoch': epoch, 'model_state': best_state, 'val_acc': best_val_acc}, out_dir / 'best_checkpoint.pth')

        print(f"Epoch {epoch}/{args.epochs} | time: {epoch_time:.2f}s | train_acc: {train_acc*100:.2f}% | val_acc: {val_acc*100:.2f}% | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")

    # restore best and test
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)
    test_loss, test_acc = evaluate(model, device, test_loader, criterion)
    print("\nBest val acc: {:.4f}".format(best_val_acc))
    print("Test acc: {:.4f}".format(test_acc))
    print("Final train acc (last epoch): {:.4f}".format(history['train_acc'][-1]))
    print("Avg epoch time: {:.3f}s".format(sum(history['epoch_time'])/len(history['epoch_time'])))

    # save final model (scriptable) for deployment reference
    final_path = Path(args.out_dir) / 'model_final.pth'
    torch.save({'model_state_dict': model.state_dict(), 'params': total_params}, final_path)
    print(f"Saved final model to {final_path}")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--out-dir", type=str, default="./output")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=128, dest="batch")
    parser.add_argument("--max-lr", type=float, default=0.05, help="max lr for OneCycleLR / initial lr for AdamW")
    parser.add_argument("--opt", type=str, choices=['sgd','adamw'], default='sgd')
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--no-amp-disable", action="store_false", dest="no_amp_disable", help="enable AMP by default; pass flag to disable")
    args = parser.parse_args()
    run(args)
