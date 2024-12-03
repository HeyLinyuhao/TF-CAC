import hashlib
import os
import urllib
import warnings
from typing import Any, Union, List
from pkg_resources import packaging

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
from tqdm import tqdm

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


if packaging.version.parse(torch.__version__) < packaging.version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px-1, interpolation=BICUBIC, max_size=n_px),
        CenterCrop(n_px),
        # _convert_image_to_rgb,
        ToTensor(),
        # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        Normalize(mean=0.5, std=0.2)
    ])

def _transform_hug(n_px):
    return Compose([
        Resize(n_px-1, interpolation=BICUBIC, max_size=n_px),
        CenterCrop(n_px),
        # _convert_image_to_rgb,
        # ToTensor(),
        # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        # Normalize(mean=0.5, std=0.2)
    ])

def _transform2(n_px):
    return Compose([
        ToPILImage(),
        Resize(n_px-1, interpolation=BICUBIC, max_size=n_px),
        CenterCrop(n_px),
        # _convert_image_to_rgb,
        ToTensor(),
        # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
