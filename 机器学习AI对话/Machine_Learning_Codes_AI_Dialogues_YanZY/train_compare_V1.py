#!/usr/bin/env python3
import argparse
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchsummary import summary

# -------------------------
# 模型定义
# -------------------------
class FCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),                      # 784
            nn.Linear(784, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.fc(x)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 为了易于计算感受野，使用 padding=1 保持空间尺寸在 conv 前后
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # 28x28 -> 28x28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                     # -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),# -> 14x14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                     # -> 7x7
        )
        # 7x7 spatial, 64 channels -> 64*7*7 flattened
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

# -------------------------
# 训练/评估工具
# -------------------------
def train_one_epoch(model, device, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start = time.time()
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
    epoch_time = time.time() - start
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
# 主流程
# -------------------------
def run_training(model_name, epochs=5, batch_size=128, lr=1e-3, use_cuda=True):
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    print(f"Device: {device}, model: {model_name}")

    # 数据
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=1000, shuffle=False, num_workers=2, pin_memory=True)

    # 模型
    if model_name == 'fc':
        model = FCNet()
        # torchsummary expects input shape consistent with model: for FC, give (784,)
        summary_input = (784,)
    elif model_name == 'cnn':
        model = SimpleCNN()
        summary_input = (1, 28, 28)
    else:
        raise ValueError("model must be 'fc' or 'cnn'")

    model = model.to(device)

    # 打印参数量（torchsummary）
    print("Model summary (parameters):")
    try:
        summary(model, summary_input, device=str(device))
    except Exception as e:
        print("torchsummary summary failed:", e)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss':[], 'train_acc':[], 'test_loss':[], 'test_acc':[], 'epoch_time':[]}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(1, epochs+1):
        train_loss, train_acc, epoch_time = train_one_epoch(model, device, train_loader, criterion, optimizer)
        test_loss, test_acc = evaluate(model, device, test_loader, criterion)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['epoch_time'].append(epoch_time)

        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch}/{epochs} | time: {epoch_time:.3f}s | train_acc: {train_acc*100:.2f}% | test_acc: {test_acc*100:.2f}% | train_loss: {train_loss:.4f} | test_loss: {test_loss:.4f}")

    # 训练完成
    model.load_state_dict(best_model_wts)
    # 计算过拟合程度（使用最后一轮或最佳模型的训练/测试准确率；这里用最后一轮）
    overfit = history['train_acc'][-1] - history['test_acc'][-1]
    print("Summary:")
    print(f"Final epoch train_acc: {history['train_acc'][-1]*100:.2f}%")
    print(f"Final epoch test_acc:  {history['test_acc'][-1]*100:.2f}%")
    print(f"Overfitting (train - test): {overfit*100:.2f}%")
    print(f"Per-epoch times (s): {['{:.3f}'.format(t) for t in history['epoch_time']]}")
    return history, model

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="both", choices=["fc","cnn","both"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    runs = []
    if args.model in ("fc","both"):
        print("\n=== Training FC model ===")
        h_fc, m_fc = run_training('fc', epochs=args.epochs, batch_size=args.batch, lr=args.lr)
        runs.append(('fc', h_fc))
    if args.model in ("cnn","both"):
        print("\n=== Training CNN model ===")
        h_cnn, m_cnn = run_training('cnn', epochs=args.epochs, batch_size=args.batch, lr=args.lr)
        runs.append(('cnn', h_cnn))

    # 打印对比表格（简单）
    print("\n== Comparison ==")
    for name, h in runs:
        final_train_acc = h['train_acc'][-1]*100
        final_test_acc  = h['test_acc'][-1]*100
        overfit = final_train_acc - final_test_acc
        avg_epoch_time = sum(h['epoch_time'])/len(h['epoch_time'])
        print(f"{name:>4} | train_acc: {final_train_acc:.2f}% | test_acc: {final_test_acc:.2f}% | overfit: {overfit:.2f}% | avg epoch time: {avg_epoch_time:.3f}s")
