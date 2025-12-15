# keyword_gsc_torchaudio.py
import os
import glob
import random
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

# ------------------ 配置 ------------------
DATA_ROOT = "./data/speech_commands_v0.02/speech_commands_v0.02.tar"  # 你的路径
SR = 16000
DURATION = 1.0
N_MELS = 40
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400
BATCH_SIZE = 64
EPOCHS = 12
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
NUM_WORKERS = 0  # Windows 下建议 0 或 1
torch.manual_seed(SEED)
random.seed(SEED)

# ------------------ 帮助函数 ------------------
def read_list_file(path: str) -> List[str]:
    if not os.path.isfile(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return lines

# ------------------ Dataset for GSC (use official split lists if present) ------------------
class GSCKeywordDataset(Dataset):
    def __init__(self, root_dir: str, file_list: Optional[List[str]] = None, training: bool = True, noise_dir: Optional[str] = None):
        """
        root_dir: path to speech_commands_v0.02 root (contains folders per label and the txt lists)
        file_list: list of relative file paths (relative to root_dir) to use for this dataset. If None, will scan all.
        training: whether to apply augmentation (noise/specaugment)
        noise_dir: optional dir of noise wavs for augmentation (can be '_background_noise_' files split into wavs)
        """
        self.root = root_dir
        self.training = training

        # labels are directories in root that contain wav files (exclude list files and special files)
        all_dirs = [d for d in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir, d))]
        # filter out special folder names that may be present (e.g., '_background_noise_')
        self.labels = [d for d in all_dirs if not d.startswith('_')]

        # map label->index
        self.label2idx = {lbl: i for i, lbl in enumerate(self.labels)}
        self.samples = []  # list of (absolute_path, label_idx)

        if file_list:
            # file_list entries in official files are relative paths like "bed/0a7c2a8d_nohash_0.wav"
            for rel in file_list:
                abspath = os.path.join(root_dir, rel)
                if os.path.isfile(abspath):
                    label = rel.split('/')[0]
                    if label in self.label2idx:
                        self.samples.append((abspath, self.label2idx[label]))
        else:
            # scan all labels
            for lbl in self.labels:
                pattern = os.path.join(root_dir, lbl, '*.wav')
                for p in glob.glob(pattern) + glob.glob(pattern.upper()):
                    self.samples.append((p, self.label2idx[lbl]))

        # load noise waveforms if provided
        self.noise_waveforms = []
        if noise_dir and os.path.isdir(noise_dir):
            for p in glob.glob(os.path.join(noise_dir, '*.wav')):
                w = load_wav(p)
                self.noise_waveforms.append(w)

        # feature transforms
        self.mel_transform = MelSpectrogram(sample_rate=SR, n_fft=N_FFT, win_length=WIN_LENGTH,
                                            hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0)
        self.db_transform = AmplitudeToDB(stype='power', top_db=80.0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, lbl = self.samples[idx]
        wav = load_wav(path)  # (1, L)
        if self.training and random.random() < 0.5:
            wav = add_noise(wav, self.noise_waveforms)
        mel = wav_to_mel_db_tensor(wav, self.mel_transform, self.db_transform)  # (n_mels, time)
        if self.training:
            mel = spec_augment(mel)
        mel = mel.unsqueeze(0)  # (1, n_mels, time)
        return mel.float(), torch.tensor(lbl, dtype=torch.long)

# ------------------ IO and feature pipeline (torchaudio) ------------------
def load_wav(path: str, sr: int = SR, duration: float = DURATION):
    waveform, orig_sr = torchaudio.load(path)  # (channels, samples)
    if orig_sr != sr:
        waveform = torchaudio.functional.resample(waveform, orig_sr, sr)
    # mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    L = int(duration * sr)
    if waveform.size(1) < L:
        pad = torch.zeros((1, L - waveform.size(1)))
        waveform = torch.cat([waveform, pad], dim=1)
    else:
        waveform = waveform[:, :L]
    return waveform

def wav_to_mel_db_tensor(waveform: torch.Tensor, mel_transform: MelSpectrogram, db_transform: AmplitudeToDB):
    # waveform: (1, L)
    S = mel_transform(waveform)  # (1, n_mels, time) or (n_mels, time)
    if S.dim() == 3:
        S = S.squeeze(0)
    mel_db = db_transform(S)  # (n_mels, time)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std(unbiased=False) + 1e-6)
    return mel_db

# ------------------ Augmentations ------------------
def add_noise(waveform: torch.Tensor, noise_waveforms: List[torch.Tensor], snr_db_min: int = 0, snr_db_max: int = 10):
    if not noise_waveforms:
        return waveform
    noise = random.choice(noise_waveforms)
    if noise.size(1) < waveform.size(1):
        pad = torch.zeros((1, waveform.size(1) - noise.size(1)))
        noise = torch.cat([noise, pad], dim=1)
    noise = noise[:, :waveform.size(1)]
    snr_db = random.uniform(snr_db_min, snr_db_max)
    sig_power = waveform.norm(p=2)
    noise_power = noise.norm(p=2)
    if noise_power == 0:
        return waveform
    scale = (sig_power / noise_power) * (10 ** (-snr_db / 20))
    return waveform + scale * noise

def spec_augment(mel: torch.Tensor, time_mask_param: int = 20, freq_mask_param: int = 8, num_time_masks: int = 1, num_freq_masks: int = 1):
    n_mels, t = mel.shape
    out = mel.clone()
    for _ in range(num_time_masks):
        tm = random.randrange(0, time_mask_param + 1)
        if tm == 0: continue
        t0 = random.randrange(0, max(1, t - tm) + 1)
        out[:, t0:t0 + tm] = 0
    for _ in range(num_freq_masks):
        fm = random.randrange(0, freq_mask_param + 1)
        if fm == 0: continue
        f0 = random.randrange(0, max(1, n_mels - fm) + 1)
        out[f0:f0 + fm, :] = 0
    return out

# ------------------ Model (narrow-kernel speech CNN) ------------------
class SpeechCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(1,5), padding=(0,2)),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2,1))
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(1,5), padding=(0,2)),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2,2))
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

# ------------------ 训练 / 评估 ------------------
def train_one_epoch(model, loader, opt, criterion):
    model.train()
    total_loss = 0.0; correct = 0; total = 0
    for x,y in loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = criterion(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item() * x.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total

def eval_model(model, loader, criterion):
    model.eval()
    total_loss = 0.0; correct = 0; total = 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total

# ------------------ 主流程 ------------------
def main():
    # 读取官方划分文件（如果存在）
    val_list = read_list_file(os.path.join(DATA_ROOT, 'validation_list.txt'))
    test_list = read_list_file(os.path.join(DATA_ROOT, 'testing_list.txt'))
    # 训练集即为所有文件减去验证和测试列表
    all_rel = []
    for lbl in sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d)) and not d.startswith('_')]):
        all_rel += [os.path.join(lbl, os.path.basename(p)) for p in glob.glob(os.path.join(DATA_ROOT, lbl, '*.wav'))]
    # but better to derive all_rel from filesystem reliably:
    all_rel = []
    for lbl in sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d)) and not d.startswith('_')]):
        for p in glob.glob(os.path.join(DATA_ROOT, lbl, '*.wav')) + glob.glob(os.path.join(DATA_ROOT, lbl, '*.WAV')):
            rel = os.path.relpath(p, DATA_ROOT).replace('\\', '/')
            all_rel.append(rel)

    # normalize list entries (they are like 'bed/0a7c...' so kept as-is)
    val_set = set(val_list)
    test_set = set(test_list)
    train_list = [r for r in all_rel if r not in val_set and r not in test_set]

    print("counts => all:", len(all_rel), "train:", len(train_list), "val:", len(val_list), "test:", len(test_list))

    # build datasets
    train_ds = GSCKeywordDataset(DATA_ROOT, file_list=train_list, training=True)
    val_ds = GSCKeywordDataset(DATA_ROOT, file_list=val_list, training=False)
    test_ds = GSCKeywordDataset(DATA_ROOT, file_list=test_list, training=False)

    print("labels:", train_ds.labels, "num_classes:", len(train_ds.labels))
    # quick sanity check
    if len(train_ds) == 0:
        raise RuntimeError("Training set is empty. Check DATA_ROOT and list files.")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)

    model = SpeechCNN(num_classes=len(train_ds.labels)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, criterion)
        val_loss, val_acc = eval_model(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{EPOCHS} - train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_acc={val_acc:.4f}")

    # final test
    test_loss, test_acc = eval_model(model, test_loader, criterion)
    print(f"Final test_acc={test_acc:.4f} test_loss={test_loss:.4f}")

    # save model and label mapping
    torch.save({'model_state': model.state_dict(), 'labels': train_ds.labels}, 'gsc_speechcnn.pt')
    print("Saved model to gsc_speechcnn.pt")

if __name__ == '__main__':
    main()
