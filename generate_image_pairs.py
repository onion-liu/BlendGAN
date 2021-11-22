import argparse
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from model import Generator
from utils import ten2cv, cv2ten
import random

seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def generate(args, g_ema, device, mean_latent, sample_style, add_weight_index):
    if args.sample_zs is not None:
        sample_zs = torch.load(args.sample_zs)
    else:
        sample_zs = None

    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
            if sample_zs is not None:
                sample_z = sample_zs[i]
            else:
                sample_z = torch.randn(1, args.latent, device=device)

            sample1, _ = g_ema([sample_z],
                               truncation=args.truncation, truncation_latent=mean_latent, return_latents=False, randomize_noise=False)
            sample2, _ = g_ema([sample_z], z_embed=sample_style, add_weight_index=add_weight_index,
                               truncation=args.truncation, truncation_latent=mean_latent, return_latents=False, randomize_noise=False)

            sample1 = ten2cv(sample1)
            sample2 = ten2cv(sample2)
            out = np.concatenate([sample1, sample2], axis=1)

            cv2.imwrite(f'{args.outdir}/{str(i).zfill(6)}.jpg', out)


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--pics', type=int, default=20, help='N_PICS')
    parser.add_argument('--truncation', type=float, default=0.75)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument('--ckpt', type=str, default='', help='path to BlendGAN checkpoint')
    parser.add_argument('--style_img', type=str, default=None, help='path to style image')
    parser.add_argument('--sample_zs', type=str, default=None)
    parser.add_argument('--add_weight_index', type=int, default=6)

    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--outdir', type=str, default="")

    args = parser.parse_args()

    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    args.latent = 512
    args.n_mlp = 8

    checkpoint = torch.load(args.ckpt)
    model_dict = checkpoint['g_ema']
    if "latent_avg" in checkpoint.keys():
        latent_avg = checkpoint["latent_avg"]
    else:
        latent_avg = None
    if "truncation" in checkpoint.keys():
        args.truncation = checkpoint["truncation"]

    print('ckpt: ', args.ckpt)
    print('truncation: ', args.truncation)

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier, load_pretrained_vgg=False
    ).to(device)
    g_ema.load_state_dict(model_dict)

    if args.truncation < 1:
        if latent_avg is not None:
            mean_latent = latent_avg
            print('### use mean_latent in ckpt["latent_avg"]')
        else:
            with torch.no_grad():
                mean_latent = g_ema.mean_latent(args.truncation_mean)
                print('### generate mean_latent with \'g_ema.mean_latent\'')
    else:
        mean_latent = None
        print('### args.truncation = 1, mean_latent is None')

    if args.style_img is not None:
        img = cv2.imread(args.style_img, 1)
        img = cv2ten(img, device)
        sample_style = g_ema.get_z_embed(img)
    else:
        sample_style = torch.randn(1, args.latent, device=device)

    generate(args, g_ema, device, mean_latent, sample_style, args.add_weight_index)

    print('Done!')
