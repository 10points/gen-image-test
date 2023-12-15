import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid
import transformers
from PIL import Image

device = "mps" if torch.backends.mps.is_available() else "cpu"
device

pipeline = AutoPipelineForInpainting.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

generator = torch.Generator(device).manual_seed(92)




init_image_path = "frame_1702462022.1263134.jpg"
mask_image_path = ""


def load_im(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image)
    return image_pil

# init_image = load_im(init_image_path)
# mask_image = load_im(mask_image_path)

# prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
# image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, generator=generator).images[0]
# make_image_grid([init_image, mask_image, image], rows=1, cols=3)
