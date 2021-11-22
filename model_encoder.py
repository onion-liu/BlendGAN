import math

from collections import namedtuple

import torch
from torch import nn
from torch.nn import functional as F

import torchvision.models.vgg as vgg

from op import fused_leaky_relu


FeatureOutput = namedtuple(
    "FeatureOutput", ["relu1", "relu2", "relu3", "relu4", "relu5"])


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


class FeatureExtractor(nn.Module):
    """Reference:
        https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
    """

    def __init__(self, load_pretrained_vgg=True):
        super(FeatureExtractor, self).__init__()
        self.vgg_layers = vgg.vgg19(pretrained=load_pretrained_vgg).features
        self.layer_name_mapping = {
            '3': "relu1",
            '8': "relu2",
            '17': "relu3",
            '26': "relu4",
            '35': "relu5",
        }

    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return FeatureOutput(**output)


class StyleEmbedder(nn.Module):
    def __init__(self, load_pretrained_vgg=True):
        super(StyleEmbedder, self).__init__()
        self.feature_extractor = FeatureExtractor(load_pretrained_vgg=load_pretrained_vgg)
        self.feature_extractor.eval()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

    def forward(self, img):
        N = img.shape[0]
        features = self.feature_extractor(self.avg_pool(img))

        grams = []
        for feature in features:
            gram = gram_matrix(feature)
            grams.append(gram.view(N, -1))
        out = torch.cat(grams, dim=1)
        return out


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )


class StyleEncoder(nn.Module):
    def __init__(
        self,
        style_dim=512,
        n_mlp=4,
        load_pretrained_vgg=True,
    ):
        super().__init__()

        self.style_dim = style_dim

        e_dim = 610304
        self.embedder = StyleEmbedder(load_pretrained_vgg=load_pretrained_vgg)

        layers = []

        layers.append(EqualLinear(e_dim, style_dim, lr_mul=1, activation='fused_lrelu'))
        for i in range(n_mlp - 2):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=1, activation='fused_lrelu'
                )
            )
        layers.append(EqualLinear(style_dim, style_dim, lr_mul=1, activation=None))
        self.embedder_mlp = nn.Sequential(*layers)

    def forward(self, image):
        z_embed = self.embedder_mlp(self.embedder(image))  # [N, 512]
        return z_embed


class Projector(nn.Module):
    def __init__(self, style_dim=512, n_mlp=4):
        super().__init__()

        layers = []
        for i in range(n_mlp - 1):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=1, activation='fused_lrelu'
                )
            )
        layers.append(EqualLinear(style_dim, style_dim, lr_mul=1, activation=None))
        self.projector = nn.Sequential(*layers)

    def forward(self, x):
        return self.projector(x)
