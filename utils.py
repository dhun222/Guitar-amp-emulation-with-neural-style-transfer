import torch
from torch import nn
import torchaudio as ta
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import librosa as lr
import matplotlib.pyplot as plt
import json

"""
Mel spectrogram candidate 1
With mel basis. Input waveform must be in the from of batch
"""

class Mel_fn_batch:
    def __init__(self, sr, n_fft, hop_size, n_mels, fmin, fmax, device='cpu', center=False):
        self.n_fft = n_fft
        self.num_mels = n_mels
        self.sampling_rate = sr
        self.hop_size = hop_size
        self.fmin = fmin
        self.fmax = fmax
        self.center = center
        self.device = device

        mel = lr.filters.mel(sr=self.sampling_rate, n_fft=self.n_fft, n_mels=self.num_mels, fmin=self.fmin, fmax=self.fmax)
        self.mel_basis = torch.from_numpy(mel).float().to(self.device)
        self.hann_window = torch.hann_window(self.n_fft).to(self.device)

    def __call__(self, y):
        '''
        input
        y : waveform. tensor [length, ]
        output
        spec : mel spectrogram of y
        '''
        if torch.min(y) < -1.:
            print('min value is ', torch.min(y))
        if torch.max(y) > 1.:
            print('max value is ', torch.max(y))

        y = torch.nn.functional.pad(y.unsqueeze(1), (int((self.n_fft-self.hop_size)/2), int((self.n_fft-self.hop_size)/2)), mode='reflect')
        y = y.squeeze(1)

        spec = torch.stft(y, self.n_fft, hop_length=self.hop_size, win_length=self.n_fft, window=self.hann_window,
                        center=self.center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
        

        spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

        spec = torch.matmul(self.mel_basis, spec)
        # torch.log(torch.clamp(spec, min=1e-5))

        return torch.transpose(spec, -2, -1)


"""
Mel spectrogram candidate 2
With librosa.feature.melspectrogram function
"""
class Mel_fn_lr:
    def __init__(self, sr, n_fft, hop_size, n_mels, fmin, fmax):
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sr = sr
        self.hop_size = hop_size
        self.fmin = fmin
        self.fmax = fmax

    def __call__(self, y):
        y = y.numpy()
        mel = lr.feature.melspectrogram(y=y, n_fft=self.n_fft, hop_length=self.hop_size, win_length=self.n_fft, sr=self.sr, n_mels=self.n_mels)
        print(mel.shape)
        mel = torch.transpose(torch.from_numpy(mel), -2, -1)
        return mel



"""
Mel spectrogram candidate 3
TorchAudio
"""
class Mel_fn_ta(nn.Module):
    def __init__(self, sr, n_fft, hop_size, n_mels, fmin, fmax, to_db=False, top_db=80):
        super().__init__()
        if to_db:
            self._fn = torch.nn.Sequential(
                MelSpectrogram(sr, n_fft, n_fft, hop_size, fmin, fmax, n_mels=n_mels), 
                AmplitudeToDB("power", top_db)
            )
        else:
            self._fn = MelSpectrogram(sr, n_fft, n_fft, hop_size, fmin, fmax, n_mels=n_mels)
        
    def forward(self, x):
        x = torch.transpose(self._fn(x), -2, -1)
        return x


def test_mel_fn(Mel_fn, wav, sr, n_fft, hop_size, n_mels, fmin, fmax, to_db=False):
    """
    Test for mel spectrogram function. Plots mel spectrogram by given function. 
    """
    mel_fn = Mel_fn(sr, n_fft, hop_size, n_mels, fmin, fmax)
    mel = mel_fn(wav.unsqueeze(0)).squeeze(0).T
    if to_db:
        mel = AmplitudeToDB('power', 80)(mel)
    mel = mel.numpy()
    
    plt.figure(figsize=(10, 4))
    lr.display.specshow(mel, y_axis='mel', sr=sr, hop_length=hop_size, x_axis='time', fmin=fmin, fmax=fmax)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

"""
for environment
"""
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def get_config(path):
    with open(path) as f:
        data = f.read()

    h_json = json.loads(data)
    h = AttrDict(h_json)
    return h


def split_mel(x, frame_size):
    """
    x : 2-D tensor. [length, n_mels]
    split x into sub tensors wiht the length of n_frame and concatenate them
    Additional dimensions such as batch have to be handled outside of this function.
    """
    n_frame = x.shape[0] // frame_size
    # Cut out the left part
    x = x[0:n_frame * frame_size, :]
    
    return x.reshape(n_frame, frame_size, x.shape[1])

def cat_mel(x):
    x = x.reshape((x.shape[0] * x.shape[1], x.shape[2]))
    return x


def load_wav(path):
    x, _ = ta.load(path, normalize=False)
    # Only uses one channel if the file is in stereo
    x = x[0] / 32768
    assert x.dtype == torch.float32
    return x

def ftime(sec):
    mil = sec % 1
    s = int(sec % 60)
    m = int(sec // 60)
    h = m // 60
    m = m % 60
    return "{:02d}h {:02d}m {:02d}.".format(h, m, s) + str.split("{:.2f}".format(mil), '.')[1] + "s"