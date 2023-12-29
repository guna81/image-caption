import numpy as np
from pretrained.generator import blip
from PIL import Image
# from custom.generator import generator

def image_caption(image):
    image = Image.open(image)
    responses = blip(image)
    return responses