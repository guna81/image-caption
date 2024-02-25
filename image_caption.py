import numpy as np
from PIL import Image
# from mymodel.inference import generate_caption
from huggingface.inference import generate_caption

def image_caption(image):
    responses = generate_caption(image)
    print('responses', responses)
    return responses