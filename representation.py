import os

import numpy as np
from skimage.io import imread


def vectorize_y(meta_data_obj):
    vector = np.zeros(4)
    vector[0] = meta_data_obj['leftPaddleY']
    vector[1] = meta_data_obj['rightPaddleY']
    vector[2] = meta_data_obj['ballX']
    vector[3] = meta_data_obj['ballY']
    return vector


def get_images(path):
    x = []
    file_paths = []
    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.png') and not filename.endswith('.predicted.png'):
                file_path = os.path.join(path, filename)
                file_paths.append(file_path)
                image = imread(file_path, as_grey=True)
                x.append(image)

        break  # no recursive walk
    x = np.array(x)
    x = x.reshape(
        (
            x.shape[0],  # number of samples
            x.shape[-1],
            x.shape[-2],
            1
        )
    )
    return x, file_paths
