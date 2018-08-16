"""
This is our directory structure:
```
data/
    train/
        pong_0.png
        pong_0.png.json
        (...)
    validation/
        pong_8000.png
        pong_8000.png.json
        (...)
```
"""

import os

import numpy as np
from keras.layers import Activation, Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.optimizers import SGD

from representation import get_images

img_width = img_height = 32
num_pixels = img_width * img_height

train_dir = os.path.join('data', 'training')
x_train, _ = get_images(train_dir)
y_train = np.array(x_train).reshape(
    (
        len(x_train),
        num_pixels
    )
)

num_training_samples = len(x_train)

input_vector_shape = x_train[0].shape
print(input_vector_shape)

print('num_train_samples', num_training_samples)

model = Sequential()

model.add(
    Conv2D(
        filters=5,
        kernel_size=3,
        strides=1,
        input_shape=input_vector_shape,
    )
)
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=5, kernel_size=3, strides=1))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(12))
model.add(Activation('selu'))
model.add(Dense(64))
model.add(Activation('selu'))
model.add(Dense(num_pixels))
model.add(Activation('sigmoid'))

model.compile(
    loss='mean_squared_error',
    optimizer='rmsprop',
    metrics=['mean_absolute_error']
)
print(model.summary())

model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=10
)

model.save('pong-autoencoder.h5')
