# keyword_cnn_torchaudio.py
import os
import random
import glob
import math
import numpy as np
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

# ---------- 配置 ----------
SR = 16000
DURATION = 1.0                # 每个样本长度（秒）
N_MELS = 40
N_FFT = 512
HOP_LENGTH = 160              # 10ms @16k
WIN_LENGTH = 400              # 25ms @16k
NUM_CLASSES = 4               # 例如：你好/再见/unknown/silence
BATCH_SIZE = 32
EPOCHS = 20
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_ROOT = 'data'            # 目录结构：data/<label>/*.wav
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ---------- 工具：加载 wav（使用 torchaudio.load） ----------
def load_wav(path, sr=SR, duration=DURATION):
    waveform, orig_sr = torchaudio.load(path)  # waveform: (channels, samples)
    if orig_sr != sr:
        waveform = torchaudio.functional.resample(waveform, orig_sr, sr)
    waveform = waveform.mean(dim=0, keepdim=True)  # 转为单通道 (1, L)
    L = int(duration * sr)
    if waveform.shape[1] < L:
        pad = torch.zeros((1, L - waveform.shape[1]))
        waveform = torch.cat([waveform, pad], dim=1)
    else:
        waveform = waveform[:, :L]
    return waveform  # Tensor (1, L)

# ---------- 特征变换：Mel Spectrogram (torchaudio) ----------
mel_transform = MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH,
    n_mels=N_MELS, power=2.0
)
db_transform = AmplitudeToDB(stype='power', top_db=80.0)

def wav_to_mel_db_tensor(waveform: torch.Tensor) -> torch.Tensor:
    # waveform: (1, L)
    S = mel_transform(waveform)            # (n_mels, time) or (1, n_mels, time) depending version
    if S.dim() == 3:
        S = S.squeeze(0)
    mel_db = db_transform(S)               # (n_mels, time)
    # 标准化（每样本）
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std(unbiased=False) + 1e-6)
    return mel_db  # Tensor (n_mels, time)

# ---------- 简单数据增强 ----------
def add_noise(waveform, noise_waveforms, snr_db_min=0, snr_db_max=10):
    if not noise_waveforms:
        return waveform
    noise = random.choice(noise_waveforms)
    # 确保 noise 长度足够
    if noise.shape[1] < waveform.shape[1]:
        pad = torch.zeros((1, waveform.shape[1] - noise.shape[1]))
        noise = torch.cat([noise, pad], dim=1)
    noise = noise[:, :waveform.shape[1]]
    # 随机 SNR
    snr_db = random.uniform(snr_db_min, snr_db_max)
    sig_power = waveform.norm(p=2)
    noise_power = noise.norm(p=2)
    if noise_power == 0:
        return waveform
    scale = (sig_power / noise_power) * (10 ** (-snr_db / 20))
    return waveform + scale * noise

def spec_augment(mel: torch.Tensor, time_mask_param=20, freq_mask_param=8, num_time_masks=1, num_freq_masks=1):
    # mel: (n_mels, time)
    n_mels, t = mel.shape
    out = mel.clone()
    for _ in range(num_time_masks):
        tm = random.randrange(0, time_mask_param+1)
        if tm == 0: continue
        t0 = random.randrange(0, max(1, t - tm) + 1)
        out[:, t0:t0+tm] = 0
    for _ in range(num_freq_masks):
        fm = random.randrange(0, freq_mask_param+1)
        if fm == 0: continue
        f0 = random.randrange(0, max(1, n_mels - fm) + 1)
        out[f0:f0+fm, :] = 0
    return out

# ---------- Dataset ----------
class KeywordDataset(Dataset):
    def __init__(self, root_dir, noise_dir: Optional[str] = None, training=True):
        self.samples = []
        self.labels = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.label2idx = {l:i for i,l in enumerate(self.labels)}
        for label in self.labels:
            for p in glob.glob(os.path.join(root_dir, label, '*.wav')):
                self.samples.append((p, self.label2idx[label]))
        self.training = training
        # 加载噪声库（可选）
        self.noise_waveforms = []
        if noise_dir and os.path.isdir(noise_dir):
            for p in glob.glob(os.path.join(noise_dir, '*.wav')):
                w = load_wav(p)
                self.noise_waveforms.append(w)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, lbl = self.samples[idx]
        wav = load_wav(path)                     # (1, L)
        if self.training and random.random() < 0.5:
            wav = add_noise(wav, self.noise_waveforms)
        mel = wav_to_mel_db_tensor(wav)          # (n_mels, time)
        if self.training:
            mel = spec_augment(mel)
        mel = mel.unsqueeze(0)                   # (1, n_mels, time)
        return mel.float(), torch.tensor(lbl, dtype=torch.long)

# ---------- 模型：语音 CNN（窄核示例） ----------
class SpeechCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 输入 (B,1,N_MELS,T)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(1,5), padding=(0,2)),  # narrow along freq/time as design
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1))
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(1,5), padding=(0,2)),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2))
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(1,5), padding=(0,2)),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ---------- 训练 / 验证 函数 ----------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total

def eval_model(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total

# ---------- 主流程 ----------
def main():
    dataset = KeywordDataset(DATA_ROOT, noise_dir=None, training=True)
    n = len(dataset)
    idxs = list(range(n))
    random.shuffle(idxs)
    split = int(n * 0.8)
    train_idx, val_idx = idxs[:split], idxs[split:]
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_dataset = KeywordDataset(DATA_ROOT, noise_dir=None, training=False)
    val_loader = DataLoader(torch.utils.data.Subset(val_dataset, val_idx),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    model = SpeechCNN(num_classes=len(dataset.labels)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = eval_model(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{EPOCHS}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
    # 保存模型
    torch.save({'model_state': model.state_dict(), 'labels': dataset.labels}, 'keyword_cnn_torchaudio.pt')

    # 推理示例
    model.eval()
    test_wav = None
    # 若要测试任意 wav，请指定 test_wav 路径
    if test_wav and os.path.isfile(test_wav):
        w = load_wav(test_wav)
        mel = wav_to_mel_db_tensor(w).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,n_mels,time)
        with torch.no_grad():
            out = model(mel)
            probs = F.softmax(out, dim=1).cpu().numpy()[0]
            idx = int(probs.argmax())
            print("Pred:", dataset.labels[idx], "Probs:", probs)

if __name__ == '__main__':
    main()
