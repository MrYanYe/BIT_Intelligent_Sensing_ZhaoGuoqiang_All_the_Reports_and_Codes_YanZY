# 检查并安装必要的库
import logger
# 在训练/打印开始之前启动日志
log_path = logger.start_logging(title="LeNet5_MNIST", folder="terminal_output")

try:
    import torch
    import torchvision
    from torchsummary import summary
except ImportError:
    # !pip install torch torchvision torchsummary
    import torch
    import torchvision
    from torchsummary import summary
 
try:
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
except ImportError:
    # !pip install matplotlib numpy scikit-learn seaborn
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
 
# 设置设备（GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
 
# %% [markdown]
# ## 1. 数据准备
 
epochs = 10         # 迭代次数
lr = 0.003          # 学习率


# %%
# 定义数据转换
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),  # LeNet-5最初是为32x32图像设计的
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))  # MNIST的均值和标准差
])
 
# 下载训练集和测试集
train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)
 
test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform
)
 
# 创建数据加载器
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
 
# %% [markdown]
# ## 2. 定义修正后的LeNet-5模型
 
# %%
class LeNet5(torch.nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 卷积层1: 输入1通道，输出6通道，5x5卷积核，padding=2保持尺寸不变
        self.conv1 = torch.nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        # 平均池化层1: 2x2窗口，步长2
        self.avg_pool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        # 卷积层2: 输入6通道，输出16通道，5x5卷积核
        self.conv2 = torch.nn.Conv2d(6, 16, kernel_size=5, stride=1)
        # 平均池化层2: 2x2窗口，步长2
        self.avg_pool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        # 卷积层3: 输入16通道，输出120通道，5x5卷积核
        self.conv3 = torch.nn.Conv2d(16, 120, kernel_size=5, stride=1)
        # 全连接层1: 输入120*2*2=480，输出84
        self.fc1 = torch.nn.Linear(120 * 2 * 2, 84)
        # 全连接层2: 输入84，输出10（对应10个数字类别）
        self.fc2 = torch.nn.Linear(84, 10)
        # 激活函数使用ReLU
        self.activation = torch.nn.ReLU()
         
    def forward(self, x):
        # 输入: 32x32x1
        x = self.activation(self.conv1(x))  # 32x32x6
        x = self.avg_pool1(x)              # 16x16x6
        x = self.activation(self.conv2(x))  # 12x12x16
        x = self.avg_pool2(x)              # 6x6x16
        x = self.activation(self.conv3(x))  # 2x2x120
        x = torch.flatten(x, 1)            # 展平为480维向量(120*2*2)
        x = self.activation(self.fc1(x))    # 84维
        x = self.fc2(x)                    # 10维输出
        return x
 
# 创建模型实例并移动到设备
model = LeNet5().to(device)
print(model)
 
# 打印模型结构摘要
summary(model, (1, 32, 32))
 
# %% [markdown]
# ## 3. 定义损失函数和优化器
 
# %%
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
 
# %% [markdown]
# ## 4. 训练和测试函数
 
# %%
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
     
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
         
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
         
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
         
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
     
    train_loss /= len(train_loader)
    accuracy = 100. * correct / total
    print(f'\nTrain set: Average loss: {train_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')
    return train_loss, accuracy
 
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
     
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
     
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')
    return test_loss, accuracy
 
# %% [markdown]
# ## 5. 训练模型并可视化
 
# %%
# epochs = 10
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
 
for epoch in range(1, epochs + 1):
    train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, epoch)
    test_loss, test_acc = test(model, device, test_loader, criterion)
     
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
 
# 可视化训练过程
plt.figure(figsize=(12, 5))
 
# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
 
# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
 
plt.tight_layout()
plt.show()
 
# %% [markdown]
# ## 6. 评估模型
 
# %%
# 获取一个测试批次
data_iter = iter(test_loader)
images, labels = next(data_iter)
images, labels = images.to(device), labels.to(device)
 
# 预测
model.eval()
with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
 
# 转换回CPU用于可视化
images = images.cpu()
labels = labels.cpu()
predicted = predicted.cpu()
 
# 显示图像和预测结果
fig, axes = plt.subplots(4, 8, figsize=(15, 8))
fig.subplots_adjust(hspace=0.5, wspace=0.5)
 
for i, ax in enumerate(axes.flat):
    image = images[i].squeeze().numpy()
    ax.imshow(image, cmap='gray')
    ax.set_title(f'True: {labels[i]}\nPred: {predicted[i]}', 
                 color='green' if labels[i] == predicted[i] else 'red')
    ax.axis('off')
 
plt.show()
 
# 计算混淆矩阵
all_preds = []
all_labels = []
 
model.eval()
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, preds = torch.max(output, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(target.cpu().numpy())
 
cm = confusion_matrix(all_labels, all_preds)
 
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# 训练、测试、可视化等全部完成后，停止日志
saved_path = logger.stop_logging()
print(f"Log saved to: {saved_path}")  # 这条只输出到控制台（因为已恢复）
