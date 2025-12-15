import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# 1. 超参数
data_root = "./data/CIFAR10"  # 指向包含 cifar-10-batches-py 的父目录
batch_size = 128
num_workers = 4
valid_ratio = 0.10
seed = 42

# 2. 变换（训练和验证/测试通常不同）
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])

# 3. 加载整个训练集（50k），从本地目录读取（download=False）
full_train_dataset = datasets.CIFAR10(root=data_root, train=True,
                                      transform=train_transform, download=False)

# 4. 计算划分长度并用 random_split 划分
total_train = len(full_train_dataset)  # 应为 50000
valid_size = int(total_train * valid_ratio)  # 5000 if valid_ratio=0.1
train_size = total_train - valid_size

torch.manual_seed(seed)  # 保证可复现的随机划分
train_dataset, valid_dataset = random_split(full_train_dataset, [train_size, valid_size])

# 注意：random_split 复制了 dataset 的 transform；若要对验证集使用不同 transform，请替换其 transform：
valid_dataset.dataset = datasets.CIFAR10(root=data_root, train=True,
                                         transform=eval_transform, download=False)
# 但要确保 valid_dataset 使用的是正确的子索引（random_split 保留了索引）

# 5. 测试集（使用 eval_transform）
test_dataset = datasets.CIFAR10(root=data_root, train=False,
                                transform=eval_transform, download=False)

# 6. DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)

# 打印大小确认
print("train samples:", len(train_dataset))
print("valid samples:", len(valid_dataset))
print("test samples:", len(test_dataset))
