#!/usr/bin/env python3
# train_cifar10_run.py
# 可运行的 CIFAR-10 训练/验证/测试脚本（data_root 已设为 "./data/CIFAR10"）

import os
import argparse
import random
import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

# -----------------------
# 模型定义（含 BatchNorm 与 Dropout）
# -----------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, dropout_prob=0.3):
        super().__init__()
        def conv_block(in_ch, out_ch, p=0.0):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout(p)
            )
        self.features = nn.Sequential(
            conv_block(3, 64, p=dropout_prob),
            conv_block(64, 128, p=dropout_prob),
            conv_block(128, 256, p=dropout_prob),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# -----------------------
# 训练 / 评估 / 工具
# -----------------------
def train_one_epoch(model, optimizer, criterion, loader, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return running_loss / total, correct / total

def evaluate(model, criterion, loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    if total == 0:
        return 0.0, 0.0, np.array([]), np.array([])
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return running_loss / total, correct / total, all_preds, all_labels

def confusion_matrix_from_preds(preds, labels, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for p, t in zip(preds, labels):
        cm[t, p] += 1
    return cm

# -----------------------
# 数据变换
# -----------------------
def get_transforms(augment, train=True):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2470, 0.2435, 0.2616))
    if train:
        if augment == "default":
            t = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
                transforms.ToTensor(),
                normalize,
            ])
        elif augment == "flip":
            t = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif augment == "auto":
            t = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor(),
                normalize,
            ])
        elif augment == "none":
            t = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        else:
            raise ValueError("Unknown augment: " + str(augment))
    else:
        t = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    return t

# -----------------------
# 主流程
# -----------------------
def main(args):
    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(os.path.join(args.work_dir, "logs"), exist_ok=True)

    train_t = get_transforms(args.augment, train=True)
    eval_t = get_transforms(args.augment, train=False)

    # load full train dataset from local folder
    full_train = datasets.CIFAR10(root=args.data_root, train=True, transform=train_t, download=False)
    total = len(full_train)
    if total != 50000:
        print(f"Warning: expected 50000 train samples but found {total}. Check data_root: {args.data_root}")

    valid_len = int(total * args.validate_split)
    train_len = total - valid_len
    torch.manual_seed(args.seed)
    train_subset, valid_subset = random_split(full_train, [train_len, valid_len])

    # make validation use eval transforms (so val is not augmented)
    full_train_eval = datasets.CIFAR10(root=args.data_root, train=True, transform=eval_t, download=False)
    valid_subset = Subset(full_train_eval, valid_subset.indices)

    test_dataset = datasets.CIFAR10(root=args.data_root, train=False, transform=eval_t, download=False)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_subset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = SimpleCNN(num_classes=10, dropout_prob=args.dropout).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_epoch = -1
    stats = defaultdict(list)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, optimizer, criterion, train_loader, device)
        val_loss, val_acc, _, _ = evaluate(model, criterion, valid_loader, device)
        scheduler.step()

        stats['epoch'].append(epoch)
        stats['train_loss'].append(train_loss)
        stats['train_acc'].append(train_acc)
        stats['val_loss'].append(val_loss)
        stats['val_acc'].append(val_acc)

        print(f"Epoch {epoch:03d} | Train loss {train_loss:.4f} acc {train_acc:.4f} | Val loss {val_loss:.4f} acc {val_acc:.4f}")

        # early stopping: minimize val_loss
        if val_loss < best_val_loss - args.early_stop_min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.work_dir, "best_model.pth"))
            print(f"  New best val loss. Model saved.")
        if epoch - best_epoch >= args.early_stop_patience and epoch > args.min_epochs:
            print(f"Early stopping at epoch {epoch}. Best epoch was {best_epoch}.")
            break

    # load best model
    best_path = os.path.join(args.work_dir, "best_model.pth")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    else:
        print("Best model not found, using final weights.")

    # test evaluation
    test_loss, test_acc, test_preds, test_labels = evaluate(model, criterion, test_loader, device)
    cm = confusion_matrix_from_preds(test_preds, test_labels, num_classes=10)

    classes = ("airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck")
    airplane_idx = 0
    bird_idx = 2
    airplane_total = int(cm[airplane_idx].sum())
    airplane_mis_as_bird = int(cm[airplane_idx, bird_idx])
    airplane_mis_rate = airplane_mis_as_bird / airplane_total if airplane_total>0 else 0.0

    bird_total = int(cm[bird_idx].sum())
    bird_mis_as_airplane = int(cm[bird_idx, airplane_idx])
    bird_mis_rate = bird_mis_as_airplane / bird_total if bird_total>0 else 0.0

    print("Test loss: {:.4f} Test acc: {:.4f}".format(test_loss, test_acc))
    print("Airplane total:", airplane_total, " mis_as_bird:", airplane_mis_as_bird, " rate:", airplane_mis_rate)
    print("Bird total:", bird_total, " mis_as_airplane:", bird_mis_as_airplane, " rate:", bird_mis_rate)

    # save logs
    logs = {
        "args": vars(args),
        "stats": {k: list(v) for k, v in stats.items()},
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "confusion_matrix": cm.tolist(),
        "airplane_mis_rate": float(airplane_mis_rate),
        "bird_mis_rate": float(bird_mis_rate)
    }
    out_fn = os.path.join(args.work_dir, "logs", f"results_{args.augment}.json")
    with open(out_fn, "w") as f:
        json.dump(logs, f, indent=2)
    print("Saved logs to", out_fn)

# -----------------------
# CLI 参数
# -----------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data/CIFAR10",
                   help="CIFAR10 parent folder that contains cifar-10-batches-py")
    p.add_argument("--work_dir", type=str, default="./work", help="where to save models/logs")
    p.add_argument("--augment", type=str, default="default",
                   choices=["default", "flip", "none", "auto"], help="augmentation strategy")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--lr_step", type=int, default=60)
    p.add_argument("--lr_gamma", type=float, default=0.2)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--validate_split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--early_stop_patience", type=int, default=15)
    p.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    p.add_argument("--min_epochs", type=int, default=30)
    args = p.parse_args()
    main(args)
