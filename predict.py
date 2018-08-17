import os

import numpy as np
from keras.models import load_model
from skimage.io import imsave

from representation import get_images

validation_dir = os.path.join('data', 'validation')
x_validation, file_paths = get_images(validation_dir)

num_validation_samples = len(x_validation)

img_width = img_height = 32
num_pixels = img_width * img_height

print('num_validation_samples', num_validation_samples)

model = load_model('pong-autoencoder.h5')

predictions = model.predict(x_validation)

for i, prediction in enumerate(predictions):
    image = prediction.reshape((img_width, img_height))
    image = np.clip(image, 0.0, 1.0)
    original_file_path = file_paths[i]
    new_file_path = original_file_path + '.predicted.png'
    imsave(new_file_path, image)
