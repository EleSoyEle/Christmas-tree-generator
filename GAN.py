import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model,Sequential
from tensorflow_addons.layers import InstanceNormalization


def upsample(filters):
    layer = Sequential()
    layer.add(Conv2DTranspose(filters,(4,4),padding='same',use_bias=False,strides=2))
    layer.add(InstanceNormalization())
    layer.add(ReLU())
    return layer


def Generator():
    input_noise = Input(shape=[100,])
    layers = [
        Dense(4*3*512),
        Reshape((4,3,512)),
        upsample(512),  #8,6,512
        upsample(512),  #16,12,512
        upsample(512),  #32,24,512
        upsample(256),  #64,48,256
        upsample(128),  #128,96,256
        upsample(64),  #256,192,64
    ]
    last = Conv2DTranspose(3,(4,4),padding='same',strides=2,activation='tanh')
    x = input_noise
    for layer in layers:
        x = layer(x)
    last = last(x)
    return Model(input_noise,last)
