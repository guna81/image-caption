import tensorflow as tf
from PIL import Image

IMAGE_SHAPE=(224, 224, 3)


def resize_image(image):
    # img = load_image(image_path)
    img = image.read()
    # img = tf.io.read_file(img)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SHAPE[:-1])
    return img