from utils import generate_image,create_model
import sys
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np
def main():
    model = create_model()
    ckpt = tf.train.Checkpoint(
        generator=model)
    ckpt.restore("checkpoints/ckpt-1").expect_partial()
    path = "results/"
    while True:
        opt = str(input("Deseas crear una imagen[y/n]: "))
        if opt.lower() == "y":
            print("Crearemos la imagen")
            print("Espera un momento")
            img = generate_image(model)
            plt.title("Imagen generada por la red")
            plt.imshow(tf.squeeze(img))
            plt.show()
            save = str(input("Deseas guardar?[y/n]:"))
            if save:
                num_images = os.listdir(path)
                cv_image = np.array(tf.squeeze(img),'float32')*255
                cv_image = cv2.cvtColor(cv_image,cv2.COLOR_BGRA2RGB)
                cv2.imwrite(path + "Img{}.png".format(len(num_images)),cv_image)
                print("Imagen guardada")
            print("Acabando proceso")

        else:
            print("Terminando programa")
            sys.exit()

if __name__ == '__main__':
    main()
