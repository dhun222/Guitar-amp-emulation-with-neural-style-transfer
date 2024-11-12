import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import argparse
from time import time
from torch.utils.tensorboard import SummaryWriter
from pairDataset import PairDataset, get_sampler, Mel_fn_bigvgan
from model import Unet
from utils import get_config, ftime
import progressbar


def train_unet(model_name, h, ckpt_dir, data_dir, valid_ratio, device, ignore_prev=False):
    """
    Train Unet with given files
    """
    device = torch.device(device)
    unet = Unet(h).to(device)
    optimizer = torch.optim.Adam(unet.parameters(), h.learning_rate, betas=[h.beta1, h.beta2])

    print(f"Model name: {model_name}")

    mel_fn = Mel_fn_bigvgan(h.sampling_rate, h.n_fft, h.hop_size, h.n_mels, h.fmin, h.fmax)

    print("Loading dataset...")
    dataset = PairDataset(data_dir, h.frame_size, mel_fn, device)
    train_sampler, valid_sampler = get_sampler(dataset, valid_ratio)
    train_loader = DataLoader(dataset, batch_size=h.batch_size, sampler=train_sampler)
    valid_loader = DataLoader(dataset, batch_size=h.batch_size, sampler=valid_sampler)

    print(f"Training dataset \t: {len(train_sampler)} frames")
    print(f"Validation dataset \t: {len(valid_sampler)} frames")


    epoch = 0
    ckpt_path = os.path.join(ckpt_dir, model_name + ".ckpt")
    b_start = 0 
    if ignore_prev:
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Creating ckpt at {ckpt_path}")
        torch.save({
            'epoch': epoch, 
            'model_state_dict': unet.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict(), 
            'b_start': 0
        }, ckpt_path)
        print("Ckpt saved.")
    else:
        if os.path.exists(ckpt_path):
            print("Loading previous training state...")
            ckpt = torch.load(ckpt_path, map_location=device)
            unet.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            epoch = ckpt['epoch']
            b_start = ckpt['b_start'] + 1
            print(f"Start from epoch {epoch}")
        else:
            os.makedirs(ckpt_dir, exist_ok=True)
            print(f"Creating ckpt at {ckpt_path}")
            torch.save({
                'epoch': epoch, 
                'model_state_dict': unet.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(), 
                'b_start': 0
            }, ckpt_path)
            print("Ckpt saved.")



    unet.train()
    sw_path = os.path.join(h.tensorboard_path, model_name)
    os.makedirs(sw_path, exist_ok=True)
    sw = SummaryWriter(sw_path)
    start_time = time()

    for e in range(epoch, h.epoch):
        print(f"Epoch {e + 1}")
        epoch_start_time = time()
        # Training
        train_err = 0
        bar = progressbar.ProgressBar(maxval=len(train_loader)).start()
        for b, (x, y, emb) in enumerate(train_loader, start = b_start):
            bar.update(b)
            optimizer.zero_grad()

            y_pred = unet(x, emb)

            loss = nn.functional.l1_loss(y, y_pred)

            loss.backward()
            optimizer.step()
            
            train_err = loss.item()
            sw.add_scalar(model_name + "/train_err", train_err, e * h.batch_size + b) 


            if (e * h.batch_size + b + 1) % h.ckpt_interval == 0:
                torch.save({
                    'epoch': e, 
                    'model_state_dict': unet.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(), 
                    'b_start': b_start
                }, ckpt_path)
                print("Checkpoint saved\n")
        # Validation
        unet.eval()
        valid_err = 0
        with torch.no_grad():
            for x, y, emb in valid_loader:
                y_pred = unet(x, emb)
                valid_err = valid_err + nn.functional.l1_loss(y, y_pred).item()
        epoch_end_time = time()
        valid_err = valid_err / len(valid_sampler)
        
        bar.finish()

        sw.add_scalar(model_name + "/valid_err", valid_err, e * h.batch_size + b)

        with open(os.path.join(sw_path, "log_" + model_name + ".txt"), 'a') as f:
            f.write(f"Epoch {e} : \n\tValidation error : {valid_err}\n\tTime : {ftime(epoch_end_time - epoch_start_time)}\n\n")

        print(f"Validation error : {valid_err}\n")
        print(f"Epoch {e + 1} took {ftime(epoch_end_time - epoch_start_time)}\n")
        
        unet.train()

        b_start = 0
    
    end_time = time()
    
    print("Train done. ")
    print(f"Time : {ftime(start_time - end_time)}")
    torch.save({
        'epoch': h.epoch, 
        'model_state_dict': unet.state_dict(), 
        'optimizer_state_dict': optimizer.state_dict(), 
    }, ckpt_path)

    sw.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--valid_ratio", default=0.02, type=float)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ignore_prev", action="store_true", default=False)
    
    a = parser.parse_args()
    print(f"Model name\t\t: {a.model_name}")
    print(f"Configuration path\t: {a.config_path}")
    print(f"Ckpt directory\t\t: {a.ckpt_dir}")
    print(f"Data path\t\t: {a.data_dir}")
    print(f"Validation ratio\t: {a.valid_ratio}")
    print(f"Device\t\t\t: {a.device}")
    print(f"Ignore previous training state: {a.ignore_prev}")
    print("Continue? y/n")
    h = get_config(a.config_path)
    if input() == 'y':
        train_unet(a.model_name, h, a.ckpt_dir, a.data_dir, a.valid_ratio, a.device, a.ignore_prev)


if __name__ == "__main__":
    main()