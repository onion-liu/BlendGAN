import torch
import tempfile
import cv2
import random
import numpy as np
from pathlib import Path
from cog import BasePredictor, Path, Input

from ffhq_dataset.gen_aligned_image import FaceAlign
from model import Generator
from psp_encoder.psp_encoders import PSPEncoder
from utils import ten2cv, cv2ten


class Predictor(BasePredictor):
    def setup(self):
        size = 1024
        latent = 512
        n_mlp = 8
        self.device = 'cuda'
        checkpoint = torch.load('pretrained_models/blendgan.pt')
        model_dict = checkpoint['g_ema']

        self.g_ema = Generator(size, latent, n_mlp, channel_multiplier=2).to(self.device)
        self.g_ema.load_state_dict(model_dict)
        self.g_ema.eval()
        self.psp_encoder = PSPEncoder('pretrained_models/psp_encoder.pt', output_size=1024).to(self.device)
        self.psp_encoder.eval()
        self.fa = FaceAlign()

    def predict(
            self,
            source: Path = Input(
                description="source facial image, it will be aligned and resized to 1024x1024 first",
            ),
            style: Path = Input(
                description="style reference facial image, it will be aligned and resized to 1024x1024 first",
            ),
    ) -> Path:
        seed = 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        add_weight_index = 6
        # face alignment
        source_img = cv2.imread(str(source))
        style_img = cv2.imread(str(style))
        source_img_crop = self.fa.get_crop_image(source_img)
        style_img_crop = self.fa.get_crop_image(style_img)
        source_img_ten = cv2ten(source_img_crop, self.device)
        style_img_ten = cv2ten(style_img_crop, self.device)
        with torch.no_grad():
            sample_style = self.g_ema.get_z_embed(style_img_ten)
            sample_in = self.psp_encoder(source_img_ten)
            img_out_ten, _ = self.g_ema([sample_in], z_embed=sample_style, add_weight_index=add_weight_index,
                                        input_is_latent=True, return_latents=False, randomize_noise=False)
            img_out = ten2cv(img_out_ten)
        out = img_out
        out_path = Path(tempfile.mkdtemp()) / "out.png"
        cv2.imwrite(str(out_path), out)
        return out_path
