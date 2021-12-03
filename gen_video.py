import argparse
import os

import cv2
import numpy as np
import torch

from model import Generator
from psp_encoder.psp_encoders import PSPEncoder
from utils import ten2cv, cv2ten

import glob
from tqdm import tqdm
import random


seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def sigmoid(x, w=1):
    return 1. / (1 + np.exp(-w * x))


def get_alphas(start=-5, end=5, step=0.5, len_tail=10):
    return [0] + [sigmoid(alpha) for alpha in np.arange(start, end, step)] + [1] * len_tail


def slide(entries, margin=32):
    """Returns a sliding reference window.
    Args:
        entries: a list containing two reference images, x_prev and x_next,
                 both of which has a shape (1, 3, H, W)
    Returns:
        canvas: output slide of shape (num_frames, 3, H*2, W+margin)
    """
    _, C, H, W = entries[0].shape
    alphas = get_alphas()
    T = len(alphas) # number of frames

    canvas = - torch.ones((T, C, H*2, W + margin))
    merged = torch.cat(entries, dim=2)  # (1, 3, H*2, W)
    for t, alpha in enumerate(alphas):
        top = int(H * (1 - alpha))  # top, bottom for canvas
        bottom = H * 2
        m_top = 0  # top, bottom for merged
        m_bottom = 2 * H - top
        canvas[t, :, top:bottom, :W] = merged[:, :, m_top:m_bottom, :]
    return canvas


def slide_one_window(entries, margin=32):
    """Returns a sliding reference window.
    Args:
        entries: a list containing two reference images, x_prev and x_next,
                 both of which has a shape (1, 3, H, W)
    Returns:
        canvas: output slide of shape (num_frames, 3, H, W+margin)
    """
    _, C, H, W = entries[0].shape
    device = entries[0].device
    alphas = get_alphas()
    T = len(alphas) # number of frames

    canvas = - torch.ones((T, C, H, W + margin)).to(device)
    merged = torch.cat(entries, dim=2)  # (1, 3, H*2, W)
    for t, alpha in enumerate(alphas):
        m_top = int(H * alpha)  # top, bottom for merged
        m_bottom = m_top + H
        canvas[t, :, :, :W] = merged[:, :, m_top:m_bottom, :]
    return canvas


def tensor2ndarray255(images):
    images = torch.clamp(images * 0.5 + 0.5, 0, 1)
    return (images.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)


@torch.no_grad()
def interpolate(args, g, sample_in, sample_style_prev, sample_style_next):
    ''' returns T x C x H x W '''
    frames_ten = []
    alphas = get_alphas()

    for alpha in alphas:
        sample_style = torch.lerp(sample_style_prev, sample_style_next, alpha)
        frame_ten, _ = g([sample_in], z_embed=sample_style, add_weight_index=args.add_weight_index,
                               input_is_latent=True, return_latents=False, randomize_noise=False)
        frames_ten.append(frame_ten)
    frames_ten = torch.cat(frames_ten)
    return frames_ten


@torch.no_grad()
def video_ref(args, g, psp_encoder, img_in_ten, img_style_tens, videoWriter):
    sample_in = psp_encoder(img_in_ten)

    img_style_ten_prev, sample_style_prev = None, None

    for idx in tqdm(range(len(img_style_tens))):
        img_style_ten_next = img_style_tens[idx]
        sample_style_next = g_ema.get_z_embed(img_style_ten_next)
        if img_style_ten_prev is None:
            img_style_ten_prev, sample_style_prev = img_style_ten_next, sample_style_next
            continue

        interpolated = interpolate(args, g, sample_in, sample_style_prev, sample_style_next)
        entries = [img_style_ten_prev, img_style_ten_next]
        slided = slide_one_window(entries, margin=0)     # [T, C, H, W)
        frames = torch.cat([img_in_ten.expand_as(interpolated), slided, interpolated], dim=3).cpu()   # [T, C, H, W*3)
        frames = tensor2ndarray255(frames)  # [T, H, W*3, C)
        for frame_idx in range(frames.shape[0]):
            frame = frames[frame_idx]
            videoWriter.write(frame[:, :, ::-1])
        img_style_ten_prev, sample_style_prev = img_style_ten_next, sample_style_next

    # append last frame 10 time
    for _ in range(10):
        videoWriter.write(frame[:, :, ::-1])


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--size', type=int, default=1024)

    parser.add_argument('--ckpt', type=str, default='', help='path to BlendGAN checkpoint')
    parser.add_argument('--psp_encoder_ckpt', type=str, default='', help='path to psp_encoder checkpoint')

    parser.add_argument('--style_img_path', type=str, default=None, help='path to style image')
    parser.add_argument('--input_img_path', type=str, default=None, help='path to input image')
    parser.add_argument('--add_weight_index', type=int, default=7)

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

    del checkpoint, model_dict

    psp_encoder = PSPEncoder(args.psp_encoder_ckpt, output_size=args.size).to(device)
    psp_encoder.eval()

    input_img_paths = sorted(glob.glob(os.path.join(args.input_img_path, '*.*')))
    style_img_paths = sorted(glob.glob(os.path.join(args.style_img_path, '*.*')))[:]

    for input_img_path in input_img_paths:
        print('process: %s' % input_img_path)

        name_in = os.path.splitext(os.path.basename(input_img_path))[0]
        img_in = cv2.imread(input_img_path, 1)
        img_in = cv2.resize(img_in, (args.size, args.size))
        img_in_ten = cv2ten(img_in, device)

        img_style_tens = []

        style_img_path_rand = random.choices(style_img_paths, k=8)
        for style_img_path in style_img_path_rand:
            name_style = os.path.splitext(os.path.basename(style_img_path))[0]
            img_style = cv2.imread(style_img_path, 1)
            img_style = cv2.resize(img_style, (args.size, args.size))
            img_style_ten = cv2ten(img_style, device)

            img_style_tens.append(img_style_ten)

        fname = f'{args.outdir}/{name_in}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        videoWriter = cv2.VideoWriter(fname, fourcc, 30, (args.size * 3, args.size))
        video_ref(args, g_ema, psp_encoder, img_in_ten, img_style_tens, videoWriter)
        videoWriter.release()
        print('save video to: %s' % fname)

    print('Done!')

