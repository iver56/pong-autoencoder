import json

from skimage.io import imread
import numpy as np
import os


def vectorize_y(meta_data_obj):
    vector = np.zeros(4)
    vector[0] = meta_data_obj['leftPaddleY']
    vector[1] = meta_data_obj['rightPaddleY']
    vector[2] = meta_data_obj['ballX']
    vector[3] = meta_data_obj['ballY']
    return vector


def rescale_image(image):
    return image / 255.0


def get_images(path):
    x = []
    y = []
    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.png'):
                file_path = os.path.join(path, filename)
                image = imread(file_path, as_grey=True)
                x.append(image)
                """
                with open(filename + '.json') as meta_data_file:
                    meta_data_obj = json.load(meta_data_file)
                print(meta_data_obj)
                vector_y = vectorize_y(meta_data_obj)
                y.append(vector_y)
                """

        break  # no recursive walk
    x = np.array(x)
    x = x.reshape(
        (
            x.shape[0],  # number of samples
            x.shape[-2] * x.shape[-1],  # width * height
        )
    )
    print(x)
    y = np.array(x)
    return x, y
