import argparse
import os

import cv2
import numpy as np
import torch

from model import Generator
from psp_encoder.psp_encoders import PSPEncoder
from utils import ten2cv, cv2ten
import glob
import random

seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--size', type=int, default=1024)

    parser.add_argument('--ckpt', type=str, default='', help='path to BlendGAN checkpoint')
    parser.add_argument('--psp_encoder_ckpt', type=str, default='', help='path to psp_encoder checkpoint')

    parser.add_argument('--style_img_path', type=str, default=None, help='path to style image')
    parser.add_argument('--input_img_path', type=str, default=None, help='path to input image')
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
    print('ckpt: ', args.ckpt)

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier, load_pretrained_vgg=False
    ).to(device)
    g_ema.load_state_dict(model_dict)
    g_ema.eval()

    psp_encoder = PSPEncoder(args.psp_encoder_ckpt, output_size=args.size).to(device)
    psp_encoder.eval()

    input_img_paths = sorted(glob.glob(os.path.join(args.input_img_path, '*.*')))
    style_img_paths = sorted(glob.glob(os.path.join(args.style_img_path, '*.*')))[:]

    num = 0

    for input_img_path in input_img_paths:
        print(num)
        num += 1

        name_in = os.path.splitext(os.path.basename(input_img_path))[0]
        img_in = cv2.imread(input_img_path, 1)
        img_in_ten = cv2ten(img_in, device)
        img_in = cv2.resize(img_in, (args.size, args.size))

        for style_img_path in style_img_paths:
            name_style = os.path.splitext(os.path.basename(style_img_path))[0]
            img_style = cv2.imread(style_img_path, 1)
            img_style_ten = cv2ten(img_style, device)
            img_style = cv2.resize(img_style, (args.size, args.size))

            with torch.no_grad():
                sample_style = g_ema.get_z_embed(img_style_ten)
                sample_in = psp_encoder(img_in_ten)
                img_out_ten, _ = g_ema([sample_in], z_embed=sample_style, add_weight_index=args.add_weight_index,
                                       input_is_latent=True, return_latents=False, randomize_noise=False)
                img_out = ten2cv(img_out_ten)
            out = np.concatenate([img_in, img_style, img_out], axis=1)
            # out = img_out
            cv2.imwrite(f'{args.outdir}/{name_in}_v_{name_style}.jpg', out)

    print('Done!')

