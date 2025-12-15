import logger
# 在训练/打印开始之前启动日志
log_path = logger.start_logging(title="MASTAR_CNN", folder="terminal_output")

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from PIL import Image
import warnings

# 忽略matplotlib警告
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# 1. 检查GPU是否可用，并选择使用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义卷积神经网络模型
class ConvNet(nn.Module):
    def __init__(self, dropout1=0.5, dropout2=0.5, num_classes=10):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=4)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1024, 1024)
        self.dropout1 = nn.Dropout(dropout1)
        self.fc2 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout(dropout2)
        self.fc3 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = self.pool1(x)
        x = nn.ReLU()(self.conv2(x))
        x = self.pool2(x)
        x = nn.ReLU()(self.conv3(x))
        x = nn.ReLU()(self.conv4(x))
        x = nn.ReLU()(self.conv5(x))
        x = self.pool3(x)
        x = torch.flatten(x, 1)
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout1(x)
        x = nn.ReLU()(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        x = nn.LogSoftmax(dim=1)(x)
        return x

# 增强的训练函数，跟踪指标
def train(model, train_loader, criterion, optimizer, epochs, validation_loader=None):
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # 验证阶段
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        if validation_loader:
            model.eval()
            with torch.no_grad():
                for images, labels in validation_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_loss = val_loss / len(validation_loader.dataset)
            val_acc = val_correct / val_total
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {epoch_loss:.4f} | "
                  f"Train Acc: {epoch_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {epoch_loss:.4f} | "
                  f"Train Acc: {epoch_acc:.4f}")
    
    return train_losses, train_accs, val_losses, val_accs

# 可视化函数 - 直接显示图形
def visualize_results(train_losses, train_accs, val_losses, val_accs, 
                     y_true, y_pred, class_names, images, true_labels, pred_labels):
    """创建并显示所有可视化图形"""
    
    # 1. 损失和准确率曲线
    plt.figure(figsize=(14, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Train Loss')
    if val_losses:
        plt.plot(val_losses, 'r-', label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, 'b-', label='Train Accuracy')
    if val_accs:
        plt.plot(val_accs, 'r-', label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show(block=False)  # 非阻塞显示
    
    # 2. 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show(block=False)
    
    # 3. 样本预测结果
    plt.figure(figsize=(15, 10))
    for i in range(min(24, len(images))):  # 显示24个样本
        ax = plt.subplot(4, 6, i + 1)
        
        # 转换为numpy数组并反标准化
        img = images[i].cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        true_class = class_names[true_labels[i]]
        pred_class = class_names[pred_labels[i]]
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        plt.title(f"True: {true_class}\nPred: {pred_class}", color=color, fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle("Sample Predictions", fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.show(block=True)  # 阻塞显示，等待用户关闭

if __name__ == '__main__':
    # ----------------------------
    # 关键超参数（统一在这里修改）
    # ----------------------------
    TRAIN_PATH = 'train'         # 训练集路径
    TEST_PATH = 'test'           # 验证/测试集路径
    BATCH_SIZE = 16              # 批大小
    EPOCHS = 32                 # 训练轮数
    LR = 5e-3                    # 学习率
    MOMENTUM = 0.9               # SGD 动量
    DROPOUT1 = 0.5               # 第一个 dropout 比例
    DROPOUT2 = 0.5               # 第二个 dropout 比例
    NUM_CLASSES = 10             # 类别数
    IMAGE_SIZE = (100, 100)      # 输入图像尺寸
    # ----------------------------
    
    # 类别名称（根据您的数据集调整）
    class_names = [str(i) for i in range(NUM_CLASSES)]
    
    # 转换数据生成器
    transform_train = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    train_dataset = datasets.ImageFolder(root=TRAIN_PATH, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    validation_dataset = datasets.ImageFolder(root=TEST_PATH, transform=transform_test)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 实例化模型
    model = ConvNet(dropout1=DROPOUT1, dropout2=DROPOUT2, num_classes=NUM_CLASSES)
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

    # 训练模型
    print("Starting training...")
    train_losses, train_accs, val_losses, val_accs = train(
        model, train_loader, criterion, optimizer, 
        epochs=EPOCHS, validation_loader=validation_loader
    )
    
    # 收集验证集预测结果用于可视化
    model.eval()
    all_preds = []
    all_labels = []
    sample_images = []
    sample_labels = []
    
    with torch.no_grad():
        for images, labels in validation_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 保存部分样本用于显示
            if len(sample_images) < 24:
                sample_images.extend(images.cpu())
                sample_labels.extend(labels.cpu())
    
    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    sample_images = torch.stack(sample_images[:24])
    sample_labels = np.array(sample_labels[:24])
    
    # 获取预测结果
    sample_preds = []
    for img in sample_images:
        img = img.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img)
            _, pred = torch.max(output, 1)
            sample_preds.append(pred.item())
    
    # 显示所有可视化结果（直接弹出窗口）
    print("\nGenerating visualizations...")
    visualize_results(
        train_losses, 
        train_accs, 
        val_losses, 
        val_accs,
        all_labels,
        all_preds,
        class_names,
        sample_images,
        sample_labels,
        sample_preds
    )
    
    # 训练、测试、可视化等全部完成后，停止日志
    saved_path = logger.stop_logging()
    print(f"Log saved to: {saved_path}")