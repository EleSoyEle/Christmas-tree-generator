import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from GAN import Generator


def generate_image(generator):
    noise = tf.random.normal([1,100])
    img = generator(noise)
    img = img*0.5 + 0.5
    return img




def create_model():
    model = Generator()
    return model
