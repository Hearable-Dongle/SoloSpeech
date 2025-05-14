import yaml
import random
import argparse
import os
import librosa
import soundfile as sf
from tqdm import tqdm
import time

import torch
import torchaudio
from torch.utils.data import DataLoader, ConcatDataset

from accelerate import Accelerator
from diffusers import DDIMScheduler

from model.solospeech.conditioners import SoloSpeech_TSR
from inference import eval_tsr
from dataset import TSRDataset
from vae_modules.autoencoder_wrapper import Autoencoder

parser = argparse.ArgumentParser()

# data loading settings
parser.add_argument('--train-csv-dir', type=str, default='')
parser.add_argument('--val-csv-dir', type=str, default='')
parser.add_argument('--base-dir', type=str, default='')
parser.add_argument('--vae-dir', type=str, default='')

parser.add_argument('--sample-rate', type=int, default=16000)
parser.add_argument('--vae-rate', type=int, default=50)
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--min-length', type=float, default=3.0)
parser.add_argument("--num-infer-steps", type=int, default=50)
# training settings
parser.add_argument("--amp", type=str, default='fp16')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--num-workers', type=int, default=16)
parser.add_argument('--num-threads', type=int, default=1)
parser.add_argument('--save-every', type=int, default=1)
parser.add_argument("--adam-epsilon", type=float, default=1e-08)

# model configs
parser.add_argument('--diffusion-config', type=str, default='./config/SoloSpeech-tsr-base.yaml')
parser.add_argument('--autoencoder-path', type=str, default='./pretrained_models/audio-vae.pt')
parser.add_argument('--resume-from', type=str, default=None, help='Path to checkpoint to resume training')

# optimization
parser.add_argument('--learning-rate', type=float, default=1e-4)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--weight-decay', type=float, default=1e-4)

# log and random seed
parser.add_argument('--random-seed', type=int, default=2024)
parser.add_argument('--log-step', type=int, default=50)
parser.add_argument('--log-dir', type=str, default='logs/')
parser.add_argument('--save-dir', type=str, default='ckpt/')


args = parser.parse_args()

with open(args.diffusion_config, 'r') as fp:
    args.diff_config = yaml.safe_load(fp)


args.v_prediction = args.diff_config["ddim"]["v_prediction"]
args.log_dir = args.log_dir.replace('log', args.diff_config["system"] + '_log')
args.save_dir = args.save_dir.replace('ckpt', args.diff_config["system"] + '_ckpt')

if os.path.exists(args.log_dir + '/audio/gt') is False:
    os.makedirs(args.log_dir + '/audio/gt', exist_ok=True)

if os.path.exists(args.save_dir) is False:
    os.makedirs(args.save_dir, exist_ok=True)

    
def masked_mse_loss(predictions, targets, mask=None):
    """
    Computes the masked mean squared error (MSE) loss for tensors of shape (batch_size, sequence_length, feature_size).
    
    Args:
        predictions (torch.Tensor): The model's predictions of shape (batch_size, sequence_length, feature_size).
        targets (torch.Tensor): The ground truth values of the same shape as predictions.
        mask (torch.Tensor): A boolean mask of shape (batch_size, sequence_length) indicating which sequences to include.
    
    Returns:
        torch.Tensor: The masked MSE loss.
    """

    if mask is not None:
        mask = mask.unsqueeze(-1).long()
        mse = (predictions - targets) ** 2
        masked_mse = mse * mask
        loss = masked_mse.sum() / mask.sum()
    else:
        mse = (predictions - targets) ** 2
        loss = mse.mean()
        
    return loss

if __name__ == '__main__':
    # Fix the random seed
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # Set device
    torch.set_num_threads(args.num_threads)
    if torch.cuda.is_available():
        args.device = 'cuda'
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        args.device = 'cpu'
    
    train_set = TSRDataset(
        csv_dir=args.train_csv_dir, 
        base_dir=args.base_dir, 
        vae_dir=args.vae_dir, 
        task="sep_noisy", 
        sample_rate=args.sample_rate, 
        vae_rate=args.vae_rate,
        n_src=2, 
        min_length=args.min_length,
        debug=args.debug,
        training=True,
    )
    train_loader = DataLoader(train_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=True, collate_fn=train_set.collate)

    # use this load for check generated audio samples
    eval_set = TSRDataset(
        csv_dir=args.val_csv_dir, 
        base_dir=args.base_dir, 
        vae_dir=args.vae_dir, 
        task="sep_noisy", 
        sample_rate=args.sample_rate, 
        vae_rate=args.vae_rate,
        n_src=2, 
        min_length=args.min_length,
        debug=True,
        training=False,
    )
    eval_loader = DataLoader(eval_set, num_workers=args.num_workers, batch_size=1, shuffle=False, pin_memory=True, collate_fn=eval_set.collate)
    # use these two loaders for benchmarks

    # use accelerator for multi-gpu training
    accelerator = Accelerator(mixed_precision=args.amp)
    
    model = SoloSpeech_TSR(
        args.diff_config['diffwrap']['UDiT'],
    )

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    
    autoencoder = Autoencoder(args.autoencoder_path, 'stable_vae', quantization_first=True)
    autoencoder.eval()
    autoencoder.to(accelerator.device)

    if args.v_prediction:
        print('v prediction')
        noise_scheduler = DDIMScheduler(**args.diff_config["ddim"]['diffusers'])
    else:
        print('noise prediction')
        noise_scheduler = DDIMScheduler(**args.diff_config["ddim"]['diffusers'])

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.learning_rate,
                                  betas=(args.beta1, args.beta2),
                                  weight_decay=args.weight_decay,
                                  eps=args.adam_epsilon,
                                  )

    if args.resume_from is not None and os.path.exists(args.resume_from):
        checkpoint = torch.load(args.resume_from, map_location='cpu')
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = checkpoint["global_step"]
        start_epoch = checkpoint["epoch"] + 1  # Continue from the next epoch
        print(f"Resuming training from checkpoint: {args.resume_from}, starting from epoch {start_epoch}.")
    else:
        global_step = 0
        start_epoch = 0
    
    model.to(accelerator.device)
    model, autoencoder, optimizer, train_loader = accelerator.prepare(model, autoencoder, optimizer, train_loader)

    losses = 0
    
    if accelerator.is_main_process:
        model_module = model.module if hasattr(model, 'module') else model
        eval_tsr(model_module, autoencoder, noise_scheduler, eval_loader, args, accelerator.device, epoch='test', ddim_steps=args.num_infer_steps, eta=0)
    accelerator.wait_for_everyone()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        for step, batch in enumerate(tqdm(train_loader)):
            # compress by vae
            mixture, target, exclude, lengths = batch['mixture_vae'], batch['source_vae'], batch['exclude_vae'], batch['length']
            mixture = mixture.to(accelerator.device)
            target = target.to(accelerator.device)
            exclude = exclude.to(accelerator.device)
            lengths = lengths.to(accelerator.device)
            
            # adding noise
            noise = torch.randn(target.shape).to(accelerator.device)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (noise.shape[0],),
                                      device=accelerator.device,).long()
            noisy_target = noise_scheduler.add_noise(target, noise, timesteps)
            # v prediction - model output
            velocity = noise_scheduler.get_velocity(target, noise, timesteps)
            # inference
            pred, pred_mask = model(x=noisy_target, timesteps=timesteps, mixture=mixture, reference=exclude, x_len=lengths)
            # backward
            if args.v_prediction:
                loss = masked_mse_loss(pred, velocity, pred_mask)
            else:
                loss = masked_mse_loss(pred, noise, pred_mask)

            is_nan = torch.isnan(loss).item()
            if not is_nan: #skip nan loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                losses += loss.item()

                if accelerator.is_main_process:
                    if global_step % args.log_step == 0:
                        n = open(args.log_dir + 'log.txt', mode='a')
                        n.write(time.asctime(time.localtime(time.time())))
                        n.write('\n')
                        n.write('Epoch: [{}][{}]    Batch: [{}][{}]    Loss: {:.6f}\n'.format(
                            epoch + 1, args.epochs, step+1, len(train_loader), losses / args.log_step))
                        n.close()
                        losses = 0.0
            else:
                torch.cuda.empty_cache()
                n = open(args.log_dir + 'log.txt', mode='a')
                n.write(time.asctime(time.localtime(time.time())))
                n.write('\n')
                n.write('Epoch: [{}][{}]    Batch: [{}][{}]  Nan  Loss\n'.format(
                    epoch + 1, args.epochs, step+1, len(train_loader)))
                n.close()

        if accelerator.is_main_process:
            model_module = model.module if hasattr(model, 'module') else model
            eval_tsr(model_module, autoencoder, noise_scheduler, eval_loader, args, accelerator.device, epoch=epoch+1, ddim_steps=args.num_infer_steps, eta=0)

            if (epoch + 1) % args.save_every == 0:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save({
                    "model": unwrapped_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                }, args.save_dir+str(epoch)+'.pt')
        accelerator.wait_for_everyone()
