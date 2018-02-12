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
from keras.layers import Activation, Dense
from keras.models import Sequential

from representation import get_images

train_dir = os.path.join('data', 'training')
x_train, _ = get_images(train_dir)
y_train = np.array(x_train)

num_training_samples = len(x_train)

img_width = img_height = 32
num_pixels = img_width * img_height

print('num_train_samples', num_training_samples)

model = Sequential()
model.add(Dense(64, input_shape=(num_pixels,)))
model.add(Activation('sigmoid'))
model.add(Dense(4))
model.add(Activation('sigmoid'))
model.add(Dense(64))
model.add(Activation('sigmoid'))
model.add(Dense(num_pixels))
model.add(Activation('sigmoid'))

model.compile(
    loss='mean_squared_error',
    optimizer='rmsprop',
    metrics=['mae']
)
print(model.summary())

model.fit(
    x_train,
    y_train,
    batch_size=64,
    nb_epoch=300
)

model.save('pong-autoencoder.h5')
