import torch
from torch import nn
import os
from utils import get_config, split_mel, cat_mel
from pairDataset import Mel_fn_bigvgan
from bigvgan.bigvgan import BigVGAN as Vocoder


"""
Unet model and helper functions
"""

class Encoder(nn.Module):
    def __init__(self, h, verbose):
        super().__init__()
        self.layers = self.makeSequential(h.encoder_list)
        self.h = h
        self.verbose = verbose

    def forward(self, x, FiLM):
        skip_list = []
        FiLM_idx = 0
        if self.verbose:
            print("\n----- Encoder -----")
        for i, module in enumerate(self.layers):
            if self.verbose:
                print(f"\nmodule # {i} : {module}")
                print(f"input : {x.shape}")
            if isinstance(module, downsample_fn):
                skip_list.append(x)
                if self.verbose:
                    print(f"skip connection output : {x.shape}")
                x = module(x)
            elif isinstance(module, FiLM_layer):
                if self.verbose:
                    print(f"FiLM idx : {FiLM_idx}")
                x = module(FiLM[FiLM_idx], x)
                FiLM_ch_idx = FiLM_ch_idx + 1
            else:
                x = module(x)
            
        if self.verbose:
            print("\n")
        
        skip_list.reverse()
        return x, skip_list

    def makeSequential(self, conf_list):
        layers = nn.ModuleList()
        prev_ch = 1
        self.FiLM_ch = []
        for module in (conf_list):
            for submodule in module:
                if submodule[0] == "conv":
                    layers.append(nn.Conv2d(prev_ch , submodule[1], submodule[2], padding=submodule[2] // 2, padding_mode="reflect"))
                    prev_ch = submodule[1]
                elif submodule[0] == "relu":
                    layers.append(nn.ReLU(inplace=True))
                elif submodule[0] == "lin":
                    layers.append(nn.Linear(prev_ch, submodule[1]))
                elif submodule[0] == "FiLM":
                    layers.append(FiLM_layer())
                    self.FiLM_ch.append(prev_ch)
                else:
                    print("Unknown module name!!")

            layers.append(downsample_fn(downsample_arg))
        
        return layers


class Decoder(nn.Module):
    def __init__(self, h, verbose):
        super().__init__()
        self.layers = self.makeSequential(h.decoder_list)
        self.verbose = verbose

    def forward(self, x, skip, FiLM):
        skip_idx= 0
        FiLM_idx = 0
        if self.verbose:
            print("\n----- Decoder -----")
        for i, module in enumerate(self.layers):
            if self.verbose:
                print(f"\nmodule # {i} : {module}")
            if isinstance(module, upsample_fn):
                if self.verbose:
                    print("input : ", x.shape)
                x = module(x)
                x = torch.cat((skip[skip_idx], x), -3)
                if self.verbose:
                    print("skip connection input : ", skip[skip_idx].shape)
                    print(f"after cat : {x.shape}")

                skip_idx = skip_idx + 1
            elif isinstance(module, FiLM_layer):
                if self.verbose:
                    print(f"FiLM idx : {FiLM_idx}")
                x = module(FiLM[FiLM_idx], x)
                FiLM_idx = FiLM_idx + 1
            else:
                x = module(x)
        if self.verbose:
            print(f"output : {x.shape}") 
        return x

    def makeSequential(self, conf_list):
        layers = nn.ModuleList()
        prev_ch = conf_list[0]
        self.FiLM_ch = []
        for module in (conf_list[1:]):
            out_ch = int(prev_ch / 2)
            layers.append(upsample_fn(prev_ch, out_ch, 2, 2))
            for submodule in module:
                if submodule[0] == "conv":
                    layers.append(nn.Conv2d(prev_ch , submodule[1], submodule[2], padding=submodule[2] // 2, padding_mode="reflect"))
                    prev_ch = submodule[1]
                elif submodule[0] == "relu":
                    layers.append(nn.ReLU(inplace=True))
                elif submodule[0] == "lin":
                    layers.append(nn.Linear(prev_ch, submodule[1]))
                elif submodule[0] == "FiLM":
                    layers.append(FiLM_layer())
                    self.FiLM_ch.append(prev_ch)
                else:
                    print("Unknown module name!!")
        
        return layers

class MidBlock(nn.Module):
    def __init__(self, h, verbose):
        super().__init__()
        self.layers = self.makeSequential(h.mid_list)
        self.verbose = verbose

    def forward(self, x, FiLM):
        if self.verbose:
            print("\n----- MidBlock -----")
        FiLM_idx = 0
        for i, module in enumerate(self.layers):
            if self.verbose:
                print(f"\nmodule # {i} : {module}")
                print(f"input : {x.shape}")
            if isinstance(module, FiLM_layer):
                if self.verbose:
                    print(f"FiLM idx : {FiLM_idx}")
                x = module(FiLM[FiLM_idx], x)
                FiLM_idx = FiLM_idx + 1
            else:
                x = module(x)

        if self.verbose:
            print("\n")
        
        return x

    def makeSequential(self, conf_list):
        layers = nn.ModuleList()
        prev_ch = conf_list[0]
        self.FiLM_ch = []
        for module in (conf_list[1:]):
            for submodule in module:
                if submodule[0] == "conv":
                    layers.append(nn.Conv2d(prev_ch , submodule[1], submodule[2], padding=submodule[2] // 2, padding_mode="reflect"))
                    prev_ch = submodule[1]
                elif submodule[0] == "relu":
                    layers.append(nn.ReLU(inplace=True))
                elif submodule[0] == "lin":
                    layers.append(nn.Linear(prev_ch, submodule[1]))
                elif submodule[0] == "FiLM":
                    layers.append(FiLM_layer())
                    self.FiLM_ch.append(prev_ch)
                else:
                    print("Unknown module name!!")
        
        return layers

"""
def stoi(string):
    if string[0] == '1' or string[0] == '2' or string[0] == '3' or string[0] == '4' or string[0] == '5' or string[0] == '6' or string[0] == '7' or string[0] == '8' or string[0] == '9' or string[0] == '0':
        return int(string)
    else:
        return string
"""
"""
def str2arg(string):
    string = string.replace(" ", "")
    string = string.replace("'", "")
    string = string.replace('"', '')
    args = str.split(string, ',')
    for i, ch in enumerate(args):
        args[i] = stoi(ch)
    return args
"""



class FiLM_layer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, FiLM, x):
        """
        FiLM : embedding input. It's slicing must be handled outside of the function. 
        x : input feature to manupulate
        """
        y = x[..., :, :, :] * FiLM[:, 0].unsqueeze(1).unsqueeze(1) + FiLM[:, 1].unsqueeze(1).unsqueeze(1)
        return y

class FiLM_gen(nn.Module):
    """
    FiLM generator
    x[..., 0] : gammas
    x[..., 1] : betas
    """
    def __init__(self, encoder_ch, mid_block_ch, decoder_ch, verbose=False):
        super().__init__()
        ch = 0
        for c in encoder_ch:
            ch = ch + c
        for c in mid_block_ch:
            ch = ch + c
        for c in decoder_ch:
            ch = ch + c
        
        self.encoder_ch = encoder_ch
        self.mid_block_ch = mid_block_ch
        self.decoder_ch = decoder_ch
        if verbose:
            print("FILM layer : ")
            print(f"encoder_ch : {encoder_ch}")
            print(f"mid_block_ch : {mid_block_ch}")
            print(f"decoder_ch : {decoder_ch}")
            print(f"ch sum : {ch}\n")
        
        self.layer = nn.Linear(4, ch * 2)

    def forward(self, x):
        x = self.layer(x)
        x = x.reshape([-1, 2])
        encoder = []
        prev_ch = 0
        for ch in self.encoder_ch:
            encoder.append(x[prev_ch:prev_ch + ch, ...])
            prev_ch = prev_ch + ch

        mid_block = []
        for ch in self.mid_block_ch:
            mid_block.append(x[prev_ch:prev_ch + ch, ...])
            prev_ch = prev_ch + ch

        decoder = []
        for ch in self.decoder_ch:
            decoder.append(x[prev_ch:prev_ch + ch, ...])
            prev_ch = prev_ch + ch
        
        return  encoder, mid_block, decoder

class Unet(nn.Module):
    def __init__(self, h, verbose=False):
        super().__init__()
        self.h = h
        global upsample_fn
        global downsample_fn
        global downsample_arg
        '''
        temp = str.split(self.h.upsample_fn, '.')
        upsample_fn = getattr(getattr(sys.modules[__name__], temp[-2]), temp[-1])
        temp = str.split(self.h.downsample_fn[0], '.')
        downsample_fn = getattr(getattr(sys.modules[__name__], temp[-2]), temp[-1])
        downsample_arg = self.h.downsample_fn[1]
        '''
        upsample_fn = torch.nn.ConvTranspose2d
        downsample_fn = torch.nn.MaxPool2d
        downsample_arg = [2, 2]
        self.encoder = Encoder(self.h, verbose)
        self.decoder = Decoder(self.h, verbose)
        self.mid_block = MidBlock(self.h, verbose)

        self.FiLM_gen = FiLM_gen(self.encoder.FiLM_ch, self.mid_block.FiLM_ch, self.decoder.FiLM_ch, verbose)
        self.modulelist = torch.nn.ModuleList([self.encoder, self.decoder, self.mid_block, self.FiLM_gen])


    def forward(self, x, emb):
        FiLM_encoder, FiLM_mid_block, FiLM_decoder = self.FiLM_gen(emb)

        # Make dimension for channel
        x = x.unsqueeze(-3)

        x, skip = self.encoder(x, FiLM_encoder)
        x = self.mid_block(x, FiLM_mid_block)
        y = self.decoder(x, skip, FiLM_decoder)

        return y.squeeze(-3)

  
class Emulator(nn.Module):
    def __init__(self, model_name, h, ckpt_path, device):
        super().__init__()
        self.h = h

        h_vocoder = get_config(h.vocoder_config_path)
        
        self.device = device

        self.unet = Unet(h).to(self.device)
        self.unet.eval()
        self.mel_fn = Mel_fn_bigvgan(h.sampling_rate, h.n_fft, h.hop_size, h.n_mels, h.fmin, h.fmax)
        self.vocoder = Vocoder(h_vocoder, use_cuda_kernel=False).to(self.device) 

        # Load ckpt for Unet
        unet_ckpt_path = os.path.join(ckpt_path, model_name + ".ckpt")
        assert os.path.exists(unet_ckpt_path), "Unet ckpt does not exist!!!"
        unet_ckpt = torch.load(unet_ckpt_path, map_location=self.device)
        self.unet.load_state_dict(unet_ckpt["model_state_dict"])
        print("unet ckpt loaded")

        # Load ckpt for vocoder
        assert os.path.exists(h.vocoder_ckpt_path), "Vocoder ckpt does not exist!!!"
        vocoder_ckpt = torch.load(h.vocoder_ckpt_path, map_location=self.device)
        self.vocoder.load_state_dict(vocoder_ckpt["generator"])
        self.vocoder.eval()
        self.vocoder.remove_weight_norm()


    def forward(self, x, emb):
        emb = emb.to(self.device)
        mel = self.mel_fn(x)
        mel = split_mel(mel, self.h.frame_size).to(self.device)
        y_mel = self.unet(mel, emb)
        y_mel = cat_mel(y_mel).T
        wav = self.vocoder(y_mel.unsqueeze(0)).squeeze(0).squeeze(0)

        return wav
    