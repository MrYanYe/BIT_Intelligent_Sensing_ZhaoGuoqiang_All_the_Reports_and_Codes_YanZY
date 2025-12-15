#!/usr/bin/env python3
"""
train_compare_bn_gap.py

支持模型:
 - fc         : 前馈全连接 784->256->128->10
 - baseline   : 原始 SimpleCNN (Conv3x3-32 -> Pool -> Conv3x3-64 -> Pool -> FC128 -> 10)
 - bn         : SimpleCNN + BatchNorm after each Conv
 - gap        : SimpleCNN with Global Average Pooling (GAP) replacing large FC
 - bn_gap     : SimpleCNN with BatchNorm + GAP + Dropout (推荐)

示例:
 python train_compare_bn_gap.py --model bn_gap --epochs 10 --batch 128 --seed 42
"""

import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchsummary import summary

# -------------------------
# 小工具：固定随机数种子
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 可选：提高可复现性，但可能影响性能
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

# -------------------------
# 模型定义
# -------------------------
class FCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x)

class SimpleCNNBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class SimpleCNN_BN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class SimpleCNN_GAP(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Flatten(),        # (N,64)
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

class SimpleCNN_BN_GAP(nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Flatten(),        # (N,64)
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

# -------------------------
# 训练 / 评估函数（含精确计时）
# -------------------------
def train_one_epoch(model, device, dataloader, criterion, optimizer, epoch_idx=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start = time.perf_counter()
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    # 精确化计时（确保 GPU 完成）
    if device.type == 'cuda':
        torch.cuda.synchronize()
    epoch_time = time.perf_counter() - start
    epoch_loss = running_loss / total
    acc = correct / total
    return epoch_loss, acc, epoch_time

def evaluate(model, device, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc

# -------------------------
# 主流程：训练并打印对比信息
# -------------------------
def run_training(model_name, epochs=10, batch_size=128, lr=1e-3, weight_decay=1e-4, seed=42, use_cuda=True):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    print(f"Device: {device} | model: {model_name} | seed: {seed}")

    # 数据集（MNIST）
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=1000, shuffle=False, num_workers=4, pin_memory=True)

    # 选择模型
    if model_name == 'fc':
        model = FCNet()
        summary_input = (784,)
    elif model_name == 'baseline':
        model = SimpleCNNBaseline()
        summary_input = (1,28,28)
    elif model_name == 'bn':
        model = SimpleCNN_BN()
        summary_input = (1,28,28)
    elif model_name == 'gap':
        model = SimpleCNN_GAP()
        summary_input = (1,28,28)
    elif model_name == 'bn_gap':
        model = SimpleCNN_BN_GAP(dropout_p=0.5)
        summary_input = (1,28,28)
    else:
        raise ValueError("model must be one of: fc, baseline, bn, gap, bn_gap")

    model = model.to(device)

    # 打印参数量和结构（torchsummary）
    print("\nModel summary (parameters):")
    try:
        # 对 FC 传入 (1,28,28) 也可行，因为 Flatten 会展平
        summary(model, summary_input, device=str(device))
    except Exception as e:
        print("torchsummary summary failed:", e)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {'train_loss':[], 'train_acc':[], 'test_loss':[], 'test_acc':[], 'epoch_time':[]}

    best_model_wts = None
    best_acc = 0.0

    for epoch in range(1, epochs+1):
        train_loss, train_acc, epoch_time = train_one_epoch(model, device, train_loader, criterion, optimizer, epoch)
        test_loss, test_acc = evaluate(model, device, test_loader, criterion)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['epoch_time'].append(epoch_time)

        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = {k:v.cpu().clone() for k,v in model.state_dict().items()}

        print(f"Epoch {epoch}/{epochs} | time: {epoch_time:.3f}s | train_acc: {train_acc*100:.2f}% | test_acc: {test_acc*100:.2f}% | train_loss: {train_loss:.4f} | test_loss: {test_loss:.4f}")

    # 恢复最佳权重（如有）
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
        model = model.to(device)

    overfit = history['train_acc'][-1] - history['test_acc'][-1]
    print("\nSummary:")
    print(f"Final epoch train_acc: {history['train_acc'][-1]*100:.2f}%")
    print(f"Final epoch test_acc:  {history['test_acc'][-1]*100:.2f}%")
    print(f"Overfitting (train - test): {overfit*100:.2f}%")
    print(f"Per-epoch times (s): {['{:.3f}'.format(t) for t in history['epoch_time']]}")
    print(f"Avg epoch time: {sum(history['epoch_time'])/len(history['epoch_time']):.3f}s")
    return history, model

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bn_gap", choices=["fc","baseline","bn","gap","bn_gap"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cuda", action="store_true", help="disable CUDA")
    args = parser.parse_args()

    run_training(model_name=args.model, epochs=args.epochs, batch_size=args.batch,
                 lr=args.lr, weight_decay=args.wd, seed=args.seed, use_cuda=(not args.no_cuda))
