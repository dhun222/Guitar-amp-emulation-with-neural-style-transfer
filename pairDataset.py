import torch
import numpy as np
import os
from torch.utils.data import Dataset, Sampler
from librosa.filters import mel as librosa_mel_fn
from utils import load_wav, split_mel

def make_file_list(data_dir, make_txt_file=False):
    # Make file list
    x_list = []
    y_list = []
    file_list = os.listdir(data_dir)
    for f in file_list:
        if f.startswith('x'):
            x_list.append(f)
        elif f.startswith('y'):
            y_list.append(f)
        
    x_list.sort()
    y_list.sort()

    emb_list = []
    for f in x_list:
        f = f.replace('.', '_')
        emb = f.split('_')
        for i, e in enumerate(emb[1:5]):
            emb[i + 1] = int(e)
        emb = torch.tensor(emb[1:5]) / 100
        emb_list.append(emb)

    if make_txt_file:
        with open(os.path.join(data_dir, "file_list.txt"), mode='w') as f:
            for i, x in enumerate(x_list):
                f.write(x + " " + y_list[i] + " " + str(emb_list[i][0].item()) + ", " + str(emb_list[i][1].item()) + ", " + str(emb_list[i][2].item()) + ", " + str(emb_list[i][3].item()) + "\n")
    
    return x_list, y_list, emb_list

# Dataset
class PairDataset(Dataset):
    """
    Dataset for Unet training
    Expect same length of audio files. 
    """
    def __init__(self, data_dir, frame_size, mel_fn, device, make_txt_file=False):
        super().__init__()
        self.device = device
        x_list, y_list, emb_list = make_file_list(data_dir, make_txt_file)

        self.data_dir = data_dir
        self.mel_fn = mel_fn
        self.frame_size = frame_size

        self.x_file_list = x_list
        self.y_file_list = y_list
        self.emb_list = emb_list
        self.n_files = len(self.x_file_list)

        self.n_frame_list = []
        self.len = 0
        for f in self.x_file_list:
            temp = load_wav(os.path.join(data_dir, f))
            temp = mel_fn(temp)
            n_frame = temp.shape[0] // frame_size
            self.n_frame_list.append(n_frame)
            self.len = self.len + n_frame
        
        self.cur_file_idx = None

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        cur = 0
        file_idx = 0
        while idx >= cur:
            cur = cur + self.n_frame_list[file_idx]
            file_idx = file_idx + 1
        file_idx = file_idx - 1
        cur = cur - self.n_frame_list[file_idx]

        if file_idx != self.cur_file_idx:
            cache_x = load_wav(os.path.join(self.data_dir, self.x_file_list[file_idx])).to(self.device)
            cache_y = load_wav(os.path.join(self.data_dir, self.y_file_list[file_idx])).to(self.device)
            cache_x = self.mel_fn(cache_x)
            cache_y = self.mel_fn(cache_y)
            cache_x = split_mel(cache_x, self.frame_size)
            cache_y = split_mel(cache_y, self.frame_size)

        return cache_x[idx - cur, ...], cache_y[idx - cur, ...], self.emb_list[file_idx].to(self.device)

class IdxSampler(Sampler):
    def __init__(self, idx):
        self.idx = idx
    
    def __iter__(self):
        return iter(self.idx)
    
    def __len__(self):
        return len(self.idx)

def get_sampler(dataset, ratio):
    n = len(dataset)
    n_valid = int(n * ratio)
    n_train = n - n_valid
    train_idx_path = os.path.join(dataset.data_dir, "train.txt")
    valid_idx_path = os.path.join(dataset.data_dir, "valid.txt")
    train_idx = []
    valid_idx = []
    if os.path.exists(train_idx_path) and os.path.exists(valid_idx_path):
        with open(train_idx_path, 'r') as f:
            for i in f:
                train_idx.append(int(i.strip()))
        with open(valid_idx_path, 'r') as f:
            for i in f:
                valid_idx.append(int(i.strip()))
    else:
        train_idx = np.random.choice(n, n_train, replace=False).tolist()
        train_idx.sort()
        with open(train_idx_path, 'w') as f:
            for i in train_idx:
                f.write(f'{i}' + '\n')

        valid_idx = [i for i in range(n) if i not in train_idx]
        with open(valid_idx_path, 'w') as f:
            for i in valid_idx:
                f.write(f'{i}' + '\n')
    
    return IdxSampler(train_idx), IdxSampler(valid_idx)

class Mel_fn_bigvgan:
    def __init__(self, sr, n_fft, hop_size, n_mels, fmin, fmax):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

        self.mel_basis_cache = {}
        self.hann_window_cache = {}

    def __call__(self, x):
        device = x.device
        key = f"{self.n_fft}_{self.n_mels}_{self.sr}_{self.hop_size}_{self.fmin}_{self.fmax}_{device}"

        if key not in self.mel_basis_cache:
            mel = librosa_mel_fn(
                sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax
            )
            self.mel_basis_cache[key] = torch.from_numpy(mel).float().to(device)
            self.hann_window_cache[key] = torch.hann_window(self.n_fft).to(device)

        mel_basis = self.mel_basis_cache[key]
        hann_window = self.hann_window_cache[key]

        padding = (self.n_fft - self.hop_size) // 2
        x = torch.nn.functional.pad(
            x.unsqueeze(0), (padding, padding), mode="reflect"
        ).squeeze(0)

        spec = torch.stft(
            x,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.n_fft,
            window=hann_window,
            center=False, 
            pad_mode="reflect", 
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)

        mel_spec = torch.matmul(mel_basis, spec)
        mel_spec = spectral_normalize_torch(mel_spec)

        return mel_spec.transpose(-1, -2)

def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)