import logger
# 在训练/打印开始之前启动日志
log_path = logger.start_logging(title="MASTAR_CNN_OPTIMIZED", folder="terminal_output")

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from PIL import Image
import warnings

# 忽略matplotlib警告
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
# 忽略学习率调度器警告
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")

# 1. 检查GPU是否可用，并选择使用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ======================== 优化后的模型定义 ========================
# 选项1：增强版ConvNet（带BatchNorm，解决过拟合）
class EnhancedConvNet(nn.Module):
    def __init__(self, dropout1=0.5, dropout2=0.5, num_classes=10):
        super(EnhancedConvNet, self).__init__()
        # 卷积层 + BatchNorm
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=5)
        self.bn1 = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=4)
        
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=1)
        
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(384)
        
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(384)
        
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=1)
        
        # 全连接层 + BatchNorm（降低复杂度）
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1024, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout1)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout2)
        
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = nn.ReLU()(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = nn.ReLU()(self.bn3(self.conv3(x)))
        x = nn.ReLU()(self.bn4(self.conv4(x)))
        x = nn.ReLU()(self.bn5(self.conv5(x)))
        x = self.pool3(x)
        
        x = torch.flatten(x, 1)
        x = nn.ReLU()(self.bn6(self.fc1(x)))
        x = self.dropout1(x)
        
        x = nn.ReLU()(self.bn7(self.fc2(x)))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x  # 移除LogSoftmax，CrossEntropyLoss自带log_softmax

# 选项2：迁移学习模型（ResNet18，适合小数据集）
class TransferLearningNet(nn.Module):
    def __init__(self, num_classes=10, freeze_backbone=True):
        super().__init__()
        # 加载预训练ResNet18
        self.backbone = models.resnet18(pretrained=True)
        # 冻结骨干网络（先训练分类头）
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        # 替换分类头
        in_feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_feat, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# ======================== 增强的训练函数（带早停+学习率调度） ========================
def train(model, train_loader, criterion, optimizer, epochs, validation_loader=None, 
          scheduler=None, patience=15, save_path="best_model.pth"):
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0  # 早停计数器
    
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
            
            # 学习率调度
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_acc)
                else:
                    scheduler.step()
            
            # 早停逻辑 + 保存最优模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, save_path)
                print(f"✅ 保存最优模型：Epoch {epoch+1}, Val Acc: {val_acc:.4f}")
            else:
                patience_counter += 1
            
            # 打印日志（获取当前学习率，替代verbose）
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs} | LR: {current_lr:.6f} | "
                  f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Patience: {patience_counter}/{patience}")
            
            # 触发早停
            if patience_counter >= patience:
                print(f"\n⚠️ 早停触发！最优Epoch: {best_epoch+1}, 最优Val Acc: {best_val_acc:.4f}")
                break
        else:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")
    
    # 加载最优模型
    if validation_loader:
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n📌 训练完成，加载最优模型（Epoch {checkpoint['epoch']+1}, Val Acc: {checkpoint['val_acc']:.4f}）")
    
    return train_losses, train_accs, val_losses, val_accs

# ======================== TTA测试增强函数 ========================
def tta_predict(model, image, tta_transforms, device):
    """测试时增强：多变换后取平均预测"""
    model.eval()
    preds = []
    with torch.no_grad():
        for transform in tta_transforms:
            # 应用变换
            img_tensor = transform(image).unsqueeze(0).to(device)
            output = model(img_tensor)
            pred = torch.softmax(output, dim=1)
            preds.append(pred)
    # 取平均后返回预测结果
    avg_pred = torch.mean(torch.stack(preds), dim=0)
    return torch.argmax(avg_pred).item(), avg_pred

# ======================== 可视化函数（保持增强版） ========================
def visualize_results(train_losses, train_accs, val_losses, val_accs, 
                     y_true, y_pred, class_names, images, true_labels, pred_labels):
    """创建并显示所有可视化图形"""
    # 1. 损失和准确率曲线
    plt.figure(figsize=(14, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Train Loss', linewidth=1.5)
    if val_losses:
        plt.plot(val_losses, 'r-', label='Validation Loss', linewidth=1.5)
    plt.title('Loss over Epochs', fontsize=12)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, 'b-', label='Train Accuracy', linewidth=1.5)
    if val_accs:
        plt.plot(val_accs, 'r-', label='Validation Accuracy', linewidth=1.5)
    plt.title('Accuracy over Epochs', fontsize=12)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=False)
    
    # 2. 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 10})
    plt.title('Confusion Matrix', fontsize=12)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show(block=False)
    
    # 3. 样本预测结果
    plt.figure(figsize=(15, 10))
    for i in range(min(24, len(images))):
        ax = plt.subplot(4, 6, i + 1)
        
        # 反标准化
        img = images[i].cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        true_class = class_names[true_labels[i]]
        pred_class = class_names[pred_labels[i]]
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        plt.title(f"True: {true_class}\nPred: {pred_class}", color=color, fontsize=9)
        plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle("Sample Predictions", fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.show(block=True)

# ======================== 主函数 ========================
if __name__ == '__main__':
    # ----------------------------
    # 关键超参数（统一修改）
    # ----------------------------
    TRAIN_PATH = 'train'               # 训练集路径
    TEST_PATH = 'test'                 # 验证/测试集路径
    BATCH_SIZE = 16                    # MX250显存较小，降回16
    EPOCHS = 100                       # 最大训练轮数（早停会提前终止）
    LR = 1e-3                          # 学习率
    MOMENTUM = 0.9                     # SGD动量
    WEIGHT_DECAY = 1e-4                # L2正则化权重衰减
    DROPOUT1 = 0.5                     # Dropout比例
    DROPOUT2 = 0.5                     # Dropout比例
    NUM_CLASSES = 10                   # 类别数
    IMAGE_SIZE = (100, 100)            # 输入图像尺寸
    PATIENCE = 3                       # 早停耐心值
    USE_TRANSFER_LEARNING = False      # 是否使用迁移学习（True/False）
    FREEZE_BACKBONE = True             # 迁移学习时是否冻结骨干网络
    # ----------------------------
    
    # 类别名称（根据数据集调整）
    class_names = [str(i) for i in range(NUM_CLASSES)]
    
    # ======================== 修复后的数据变换（核心修正） ========================
    # 训练集：强数据增强（RandomErasing移到ToTensor之后）
    transform_train = transforms.Compose([
        transforms.Resize((int(IMAGE_SIZE[0]*1.1), int(IMAGE_SIZE[1]*1.1))),  # 先放大
        transforms.RandomCrop(IMAGE_SIZE),                                   # 随机裁剪
        transforms.RandomHorizontalFlip(p=0.5),                               # 随机水平翻转
        transforms.RandomVerticalFlip(p=0.2),                                 # 随机垂直翻转
        transforms.RandomRotation(degrees=15),                                # 随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1), # 色彩抖动
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),             # 随机高斯模糊
        transforms.Resize(IMAGE_SIZE),
        transforms.Grayscale(num_output_channels=3),                          # 转3通道适配预训练模型
        transforms.ToTensor(),                                                # 先转Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))                    # 后做RandomErasing（修正核心）
    ])

    # 测试集：仅基础变换（无增强）
    transform_test = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # ======================== 定义TTA变换（测试增强） ========================
    tta_transforms = [
        # 变换1：原图
        transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        # 变换2：水平翻转
        transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        # 变换3：小角度旋转
        transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomRotation(degrees=10),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    ]

    # ======================== 加载数据集 ========================
    train_dataset = datasets.ImageFolder(root=TRAIN_PATH, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)  # Windows下num_workers=0

    validation_dataset = datasets.ImageFolder(root=TEST_PATH, transform=transform_test)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ======================== 实例化模型 ========================
    if USE_TRANSFER_LEARNING:
        model = TransferLearningNet(num_classes=NUM_CLASSES, freeze_backbone=FREEZE_BACKBONE)
        print("📌 使用迁移学习模型（ResNet18）")
    else:
        model = EnhancedConvNet(dropout1=DROPOUT1, dropout2=DROPOUT2, num_classes=NUM_CLASSES)
        print("📌 使用增强版ConvNet模型")
    
    model = model.to(device)

    # ======================== 定义损失函数和优化器 ========================
    criterion = nn.CrossEntropyLoss()  # 移除NLLLoss，直接用CrossEntropyLoss
    
    # 优化器（带权重衰减）
    if USE_TRANSFER_LEARNING and FREEZE_BACKBONE:
        # 迁移学习冻结阶段：仅优化分类头，用较大LR
        optimizer = optim.SGD(
            model.backbone.fc.parameters(),  # 仅训练分类头
            lr=LR, 
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY
        )
    else:
        # 普通训练/解冻后：训练全部参数
        optimizer = optim.SGD(
            model.parameters(), 
            lr=LR, 
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY
        )
    
    # 学习率调度器（移除verbose，解决警告）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5
    )

    # ======================== 开始训练 ========================
    print("\n========== 开始训练 ==========")
    train_losses, train_accs, val_losses, val_accs = train(
        model, train_loader, criterion, optimizer, 
        epochs=EPOCHS, validation_loader=validation_loader,
        scheduler=scheduler, patience=PATIENCE
    )
    
    # ======================== TTA预测（提升测试准确率） ========================
    print("\n========== 开始TTA预测 ==========")
    model.eval()
    all_preds = []
    all_labels = []
    sample_images = []
    sample_true_labels = []
    sample_pred_labels = []
    
    with torch.no_grad():
        for images, labels in validation_loader:
            # 收集样本图像（用于可视化）
            if len(sample_images) < 24:
                sample_images.extend(images.cpu())
                sample_true_labels.extend(labels.cpu().numpy())
            
            # TTA预测
            for img, lbl in zip(images, labels):
                pred_label, _ = tta_predict(model, img, tta_transforms, device)
                all_preds.append(pred_label)
                all_labels.append(lbl.item())
    
    # 处理样本预测结果
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    sample_images = torch.stack(sample_images[:24])
    sample_true_labels = np.array(sample_true_labels[:24])
    
    # 样本的TTA预测
    sample_pred_labels = []
    for img in sample_images:
        pred_label, _ = tta_predict(model, img, tta_transforms, device)
        sample_pred_labels.append(pred_label)
    
    # 计算最终准确率
    final_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\n🎯 TTA后最终验证准确率: {final_acc:.4f}")

    # ======================== 可视化结果 ========================
    print("\n========== 生成可视化结果 ==========")
    visualize_results(
        train_losses, train_accs, val_losses, val_accs,
        all_labels, all_preds, class_names,
        sample_images, sample_true_labels, sample_pred_labels
    )
    
    # ======================== 停止日志 ========================
    saved_path = logger.stop_logging()
    print(f"\n📝 日志已保存至: {saved_path}")
    print("✅ 所有任务完成！")