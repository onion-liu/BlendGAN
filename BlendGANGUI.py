# BlendGAN GUI for jupyter notebook

import ipywidgets as widgets
import numpy as np
import cv2
import PIL.Image as Image

from utils import ten2cv, cv2ten

import torch

print('operator compiling, wait a moment ...')

from model import Generator
from psp_encoder.psp_encoders import PSPEncoder

from ffhq_dataset.gen_aligned_image import FaceAlign

from IPython.display import display, clear_output
import random


seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def scale_image(img, scale_factor):
    h, w, _ = img.shape
    img = cv2.resize(img, (w // scale_factor, h // scale_factor))
    return img


class BlendGANGUI:
    def __init__(self, ckpt_path, psp_encoder_path, img_size=1024, truncation=0.75, scale_factor=4, device='cuda'):
        self.img_size = img_size
        self.scale_factor = scale_factor
        self.truncation = truncation
        self.device = device

        self.init_model(ckpt_path, psp_encoder_path, init_latent_avg=True)

        self.gen_latent_face()
        self.gen_latent_style()

        self.add_weight_index = 6  # [0, 18]

        self.fa = FaceAlign()

        self.uploaded_face_sample = None
        self.uploaded_style_sample = None

        self.init_gui()

        self.show_face_img()
        self.show_style_img()
        self.show_inter_img()

    def init_model(self, ckpt_path, psp_encoder_path, init_latent_avg=True):
        self.ckpt_path = ckpt_path
        self.psp_encoder_path = psp_encoder_path

        self.model = Generator(self.img_size, 512, 8, channel_multiplier=2, load_pretrained_vgg=False).to(self.device)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device)['g_ema'])
        self.model.eval()

        self.psp_encoder = PSPEncoder(psp_encoder_path, output_size=self.img_size).to(self.device)
        self.psp_encoder.eval()

        if init_latent_avg:
            self.latent_avg = self.model.mean_latent(4096)

        torch.cuda.empty_cache()

    def init_gui(self):
        clear_output(wait=True)

        self.show_face = widgets.Output()
        self.show_style = widgets.Output()
        self.show_inter = widgets.Output()
        ui_show_results = widgets.HBox([self.show_style, self.show_inter, self.show_face])


        self.show_upload_face_img = widgets.Output()
        self.show_upload_style_img = widgets.Output()

        self.btn_upload_face_widgets = widgets.FileUpload(description="upload face image", multiple=False, layout=widgets.Layout(width='auto'))
        self.btn_run_upload_face_widgets = widgets.Button(description="change face image", tooltip='change face image', layout=widgets.Layout(width='auto'))
        self.btn_run_upload_face_widgets.on_click(self.btn_click_run_upload_face)

        self.btn_upload_style_widgets = widgets.FileUpload(description="upload style image", multiple=False, layout=widgets.Layout(width='auto'))
        self.btn_run_upload_style_widgets = widgets.Button(description="change style image", tooltip='change style image', layout=widgets.Layout(width='auto'))
        self.btn_run_upload_style_widgets.on_click(self.btn_click_run_upload_style)
        ui_upload = widgets.HBox([widgets.VBox([self.btn_upload_style_widgets, self.btn_run_upload_style_widgets]),
                                  self.show_upload_style_img,
                                  widgets.VBox([self.btn_upload_face_widgets, self.btn_run_upload_face_widgets]),
                                  self.show_upload_face_img])


        self.index_widgets = widgets.IntSlider(description='value i', min=0, max=18, step=1, value=self.add_weight_index)
        self.index_widgets.observe(self.index_value_change)

        self.btn_random_latent_face_widgets = widgets.Button(description="random face latent", tooltip='random face latent')
        self.btn_random_latent_face_widgets.on_click(self.btn_click_random_latent_face)

        self.btn_random_latent_style_widgets = widgets.Button(description="random style latent", tooltip='random style latent')
        self.btn_random_latent_style_widgets.on_click(self.btn_click_random_latent_style)
        ui_random = widgets.HBox([self.btn_random_latent_style_widgets, self.index_widgets, self.btn_random_latent_face_widgets])

        self.ui_global = widgets.VBox([ui_upload, ui_random, ui_show_results])
        display(self.ui_global)

    def show_face_img(self):
        if self.uploaded_face_sample is not None:
            truncation = 1
            input_is_latent = True
        else:
            truncation = self.truncation
            input_is_latent = False

        with torch.no_grad():
            sample_face, _ = self.model([self.latent_face],
                                    truncation=truncation, truncation_latent=self.latent_avg,
                                    input_is_latent=input_is_latent, return_latents=False, randomize_noise=False)

        self.sample_face_arr = scale_image(ten2cv(sample_face, bgr=False), scale_factor=self.scale_factor)

        with self.show_face:
            clear_output(wait=True)
            display(Image.fromarray(self.sample_face_arr))

    def show_style_img(self):
        if self.uploaded_face_sample is not None:
            truncation = 1
            input_is_latent = True
        else:
            truncation = self.truncation
            input_is_latent = False

        with torch.no_grad():
            sample_style, _ = self.model([self.latent_face], z_embed=self.latent_style,
                                         truncation=truncation, truncation_latent=self.latent_avg,
                                         input_is_latent=input_is_latent, return_latents=False, randomize_noise=False)

        self.sample_style_arr = scale_image(ten2cv(sample_style, bgr=False), scale_factor=self.scale_factor)

        with self.show_style:
            clear_output(wait=True)
            display(Image.fromarray(self.sample_style_arr))

    def show_inter_img(self):
        if self.uploaded_face_sample is not None:
            truncation = 1
            input_is_latent = True
        else:
            truncation = self.truncation
            input_is_latent = False

        with torch.no_grad():
            sample_inter, _ = self.model([self.latent_face], z_embed=self.latent_style, add_weight_index=self.add_weight_index,
                                         truncation=truncation, truncation_latent=self.latent_avg,
                                         input_is_latent=input_is_latent, return_latents=False, randomize_noise=False)

        self.sample_inter_arr = scale_image(ten2cv(sample_inter, bgr=False), scale_factor=self.scale_factor)

        with self.show_inter:
            clear_output(wait=True)
            display(Image.fromarray(self.sample_inter_arr))

    def gen_latent_face(self):
        self.latent_face = torch.randn(1, 512, device=self.device)

    def gen_latent_style(self):
        self.latent_style = torch.randn(1, 512, device=self.device)

    def refresh_latent_face(self):
        self.uploaded_face_sample = None
        with self.show_upload_face_img:
            clear_output(wait=False)
        self.gen_latent_face()
        self.show_face_img()
        self.show_inter_img()
        self.show_style_img()

    def refresh_latent_style(self):
        self.uploaded_style_sample = None
        with self.show_upload_style_img:
            clear_output(wait=False)
        self.gen_latent_style()
        self.show_style_img()
        self.show_inter_img()

    def btn_click_random_latent_face(self, sender):
        self.refresh_latent_face()

    def btn_click_random_latent_style(self, sender):
        self.refresh_latent_style()

    def index_value_change(self, sender):
        self.add_weight_index = self.index_widgets.value
        self.show_inter_img()

    def get_upload_image(self, upload_widgets):
        if len(upload_widgets.data) == 0:
            return None
        rawdata = upload_widgets.data[0]
        imgstring = np.asarray(bytearray(rawdata), dtype="uint8")
        image = cv2.imdecode(imgstring, cv2.IMREAD_COLOR)
        h, w, c = image.shape
        if h == w and h in [256, 512, 1024]:    # for aligned image
            img_crop = image
        else:
            img_crop = self.fa.get_crop_image(image)

        if img_crop is None:
            return None

        img_crop = cv2.resize(img_crop, (self.img_size, self.img_size))
        return img_crop

    def btn_click_run_upload_face(self, sender):
        self.uploaded_face_sample = self.get_upload_image(self.btn_upload_face_widgets)
        if self.uploaded_face_sample is None:
            return None
        uploaded_face_sample_ten = cv2ten(self.uploaded_face_sample, self.device)
        self.latent_face = self.psp_encoder(uploaded_face_sample_ten)

        with self.show_upload_face_img:
            clear_output(wait=True)
            uploaded_face_sample_arr = scale_image(self.uploaded_face_sample, scale_factor=self.scale_factor)[:, :, ::-1]
            display(Image.fromarray(uploaded_face_sample_arr))

        self.show_face_img()
        self.show_inter_img()
        self.show_style_img()

    def btn_click_run_upload_style(self, sender):
        self.uploaded_style_sample = self.get_upload_image(self.btn_upload_style_widgets)
        if self.uploaded_style_sample is None:
            return None
        uploaded_style_sample_ten = cv2ten(self.uploaded_style_sample, self.device)
        self.latent_style = self.model.get_z_embed(uploaded_style_sample_ten)

        with self.show_upload_style_img:
            clear_output(wait=True)
            uploaded_style_sample_arr = scale_image(self.uploaded_style_sample, scale_factor=self.scale_factor)[:, :, ::-1]
            display(Image.fromarray(uploaded_style_sample_arr))

        self.show_inter_img()
        self.show_style_img()
