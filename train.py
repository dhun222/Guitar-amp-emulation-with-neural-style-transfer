import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import argparse
from time import time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pairDataset import PairDataset, get_sampler, Mel_fn_bigvgan
from model import Unet
from utils import get_config, ftime


def train_unet(a, h):
    """
    Train Unet with given files
    """
    device = torch.device(a.device)
    torch.manual_seed(h.seed)
    torch.cuda.manual_seed(h.seed)
    torch.cuda.manual_seed_all(h.seed)
    unet = Unet(h).to(device)
    optimizer = torch.optim.Adam(unet.parameters(), h.learning_rate, betas=[h.beta1, h.beta2])

    print(f"Model name: {a.model_name}")

    mel_fn = Mel_fn_bigvgan(h.sampling_rate, h.n_fft, h.hop_size, h.n_mels, h.fmin, h.fmax)

    print("Loading dataset...")
    dataset = PairDataset(a.data_dir, h.frame_size, mel_fn, device)
    train_sampler, valid_sampler = get_sampler(dataset, a.valid_ratio)
    train_loader = DataLoader(dataset, batch_size=a.batch_size, sampler=train_sampler, num_workers=a.num_workers, pin_memory=True)
    valid_loader = DataLoader(dataset, batch_size=a.batch_size, sampler=valid_sampler, num_workers=a.num_workers, pin_memory=True)

    print(f"Training dataset \t: {len(train_sampler)} frames")
    print(f"Validation dataset \t: {len(valid_sampler)} frames")

    unet.train()
    sw_path = os.path.join(a.log_dir, a.model_name)
    os.makedirs(sw_path, exist_ok=True)
    sw = SummaryWriter(sw_path)

    epoch = 0
    ckpt_path = os.path.join(a.ckpt_dir, a.model_name + ".ckpt")
    elapsed_time = 0
    if a.ignore_prev:
        os.makedirs(a.ckpt_dir, exist_ok=True)
        print(f"Creating ckpt at {ckpt_path}")
        torch.save({
            'epoch': epoch, 
            'model_state_dict': unet.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict(), 
            'elapsed_time': elapsed_time
        }, ckpt_path)
        print("Ckpt saved.")
    else:
        if os.path.exists(ckpt_path):
            print("Loading previous training state...")
            ckpt = torch.load(ckpt_path, map_location=device)
            unet.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            epoch = ckpt['epoch']
            elapsed_time = ckpt['elapsed_time']
        else:
            os.makedirs(a.ckpt_dir, exist_ok=True)
            print(f"Creating ckpt at {ckpt_path}")
            torch.save({
                'epoch': epoch, 
                'model_state_dict': unet.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(), 
                'elapsed_time': elapsed_time
            }, ckpt_path)
            print("Ckpt saved.")
    
    start_time = time() - elapsed_time

    # Repeat epochs
    for e in range(epoch, a.epoch):
        epoch_start_time = time()
        # Training
        train_err = 0
        for b, (x, y, emb) in enumerate(tqdm(train_loader, desc=f"Epoch {e}", mininterval=3)):
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            emb = emb.to(device)

            y_pred = unet(x, emb)

            loss = nn.functional.l1_loss(y, y_pred)

            loss.backward()
            optimizer.step()

            train_err = train_err + loss.item()
            
        # Save ckpt
        torch.save({
            'epoch': e, 
            'model_state_dict': unet.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict(), 
            'elapsed_time': time() - start_time
        }, ckpt_path)
        print("Checkpoint saved")

        # Validation
        unet.eval()
        valid_err = 0
        with torch.no_grad():
            for x, y, emb in valid_loader:
                x = x.to(device)
                y = y.to(device)
                emb = emb.to(device)
                y_pred = unet(x, emb)
                valid_err = valid_err + nn.functional.l1_loss(y, y_pred).item()

        valid_err = valid_err / len(valid_sampler)
        train_err = train_err / len(train_sampler)
        
        sw.add_scalar(a.model_name + "/train_err/epoch", train_err, e)
        sw.add_scalar(a.model_name + "/valid_err/epoch", valid_err, e)

        epoch_end_time = time()
        print(f"took {ftime(epoch_end_time - epoch_start_time)}")
        print(f"Training error : {train_err}")
        print(f"Validation error : {valid_err}\n")

        unet.train()

    end_time = time()
    
    print("Train done. ")
    print(f"Time : {ftime(end_time - start_time)}")
    torch.save({
        'epoch': a.epoch, 
        'model_state_dict': unet.state_dict(), 
        'optimizer_state_dict': optimizer.state_dict(), 
        'elapsed_time': end_time - start_time
    }, ckpt_path)

    sw.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--valid_ratio", default=0.02, type=float)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ignore_prev", action="store_true", default=False)
    
    a = parser.parse_args()
    print(f"Model name\t\t: {a.model_name}")
    print(f"Configuration path\t: {a.config_path}")
    print(f"Ckpt directory\t\t: {a.ckpt_dir}")
    print(f"Data path\t\t: {a.data_dir}")
    print(f"Log directory\t\t: {a.log_dir}")
    print(f"Validation ratio\t: {a.valid_ratio}")
    print(f"Epoch\t\t\t: {a.epoch}")
    print(f"Batch size\t\t: {a.batch_size}")
    print(f"num_workers\t\t: {a.num_workers}")
    print(f"Device\t\t\t: {a.device}")
    print(f"Ignore previous\ntraining state\t\t: {a.ignore_prev}")
    print("Continue? y/n")
    h = get_config(a.config_path)
    if input() == 'y':
        train_unet(a, h)


if __name__ == "__main__":
    main()