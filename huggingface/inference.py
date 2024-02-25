from .models import blip

def generate_caption(image):
    return blip(image)