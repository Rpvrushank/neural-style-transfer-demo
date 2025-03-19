import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import os

# Function to load and preprocess images
def load_img(path_to_img):
    max_dim = 512  # Keep images manageable
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]  # Add batch dimension
    return img

# Function to save the result
def save_img(tensor, file_name):
    tensor = tf.squeeze(tensor)  # Remove batch dimension
    tensor = tf.clip_by_value(tensor, 0, 1)  # Ensure values are 0-1
    tensor = tf.image.convert_image_dtype(tensor, tf.uint8)
    img = Image.fromarray(tensor.numpy())
    img.save(file_name)

# Load content and style images
content_path = 'content.jpg'
style_path = 'style.jpg'
content_image = load_img(content_path)
style_image = load_img(style_path)

# Load the pre-trained style transfer model from TensorFlow Hub
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Run style transfer
stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

# Save the output
save_img(stylized_image, 'output.jpg')
print("Done! Check 'output.jpg' in your folder.")