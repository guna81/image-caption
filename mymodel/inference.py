import tensorflow as tf

from .utils import resize_image
from .model import Captioner, tokenizer, mobilenet, process_datasets

_, _, output_layer = process_datasets()

# model = Captioner(tokenizer, mobilenet, output_layer, num_layers=1, units=256, max_length=50, num_heads=1, dropout_rate=0.1)
model = Captioner(tokenizer, feature_extractor=mobilenet, output_layer=output_layer,
                  units=256, dropout_rate=0.5, num_layers=2, num_heads=2)
model.load_weights("mymodel/models/model_weights")

def generate_caption(image):
    image = resize_image(image)
    result = model.simple_gen(image, temperature=0.0)
    print('result', result)
    return result