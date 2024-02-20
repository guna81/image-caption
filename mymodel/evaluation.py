import einops
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# from process_data import load_image
from .model import Captioner, tokenizer, mobilenet, output_layer, image, load_image
from .visualize import plot_attention_maps

# load saved model
# model = tf.keras.models.load_model("mymodel/models/model", custom_objects={
#     'Captioner': lambda tokenizer, mobilenet, output_layer: Captioner(
#         tokenizer, mobilenet, output_layer, num_layers=1,
#         units=256, max_length=50, num_heads=1, dropout_rate=0.1
#     )
# })

# create model and load weights
model = Captioner(tokenizer, mobilenet, output_layer, num_layers=1, units=256, max_length=50, num_heads=1, dropout_rate=0.1)

model.load_weights("mymodel/models/model_weights")

print(model)

result = model.simple_gen(image, temperature=0.0)
print('result', result)

str_tokens = result.split()
str_tokens.append('[END]')

attn_maps = [layer.last_attention_scores for layer in model.decoder_layers]
[map.shape for map in attn_maps]

attention_maps = tf.concat(attn_maps, axis=0)
attention_maps = einops.reduce(
    attention_maps,
    'batch heads sequence (height width) -> sequence height width',
    height=7, width=7,
    reduction='mean')

einops.reduce(attention_maps, 'sequence height width -> sequence', reduction='sum')

plot_attention_maps(image/255, str_tokens, attention_maps)

@Captioner.add_method
def run_and_show_attention(self, image, temperature=0.0):
  result_txt = self.simple_gen(image, temperature)
  str_tokens = result_txt.split()
  str_tokens.append('[END]')

  attention_maps = [layer.last_attention_scores for layer in self.decoder_layers]
  attention_maps = tf.concat(attention_maps, axis=0)
  attention_maps = einops.reduce(
      attention_maps,
      'batch heads sequence (height width) -> sequence height width',
      height=7, width=7,
      reduction='mean')

  plot_attention_maps(image/255, str_tokens, attention_maps)
  t = plt.suptitle(result_txt)
  t.set_y(1.05)


run_and_show_attention(model, image)


image_url = 'https://tensorflow.org/images/bedroom_hrnet_tutorial.jpg'
image_path = tf.keras.utils.get_file(origin=image_url)
image = load_image(image_path)

run_and_show_attention(model, image)