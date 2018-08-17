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
from keras.layers import (
    Activation,
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    Deconvolution2D,
)
from keras.layers import K
from keras.models import Sequential

from representation import get_images

train_dir = os.path.join("data", "training")
x_train, _ = get_images(train_dir)

y_train = np.array(x_train)
y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], 1))

num_training_samples = len(x_train)

input_vector_shape = x_train[0].shape
print(input_vector_shape)

print("num_train_samples", num_training_samples)


def clipped_mean_squared_error(y_true, y_pred):
    """Clip to [0, 1] before calculating the mean squared error."""
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.mean(K.square(y_pred - y_true), axis=-1)


model = Sequential()

model.add(
    Conv2D(
        filters=5,
        kernel_size=3,
        strides=1,
        input_shape=input_vector_shape,
        padding="same",
    )
)

model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=5, kernel_size=3, strides=1, padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(
    Deconvolution2D(
        1, kernel_size=(4, 4), strides=(2, 2), padding="same"
    )
)
model.add(Activation('sigmoid'))

model.compile(
    loss='mean_squared_error', optimizer="rmsprop", metrics=["mean_absolute_error"]
)
print(model.summary())

model.fit(x_train, y_train, batch_size=32, epochs=10)

model.save("pong-autoencoder.h5")
