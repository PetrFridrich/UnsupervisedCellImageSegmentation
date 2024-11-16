import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2


def load_image(path):

    return cv2.imread(path)


def load_label(path):

    return np.loadtxt(path, dtype=np.uint8)


def save_image(img, output_path):

    plt.imsave(output_path, img)

    return None


def resize_img(img):

    WIDTH = 500
    HEIGHT = 500

    img = cv2.resize(img, (WIDTH, HEIGHT))

    return img


def img_to_X(img):

    X = pd.DataFrame()

    X['B'] = img[:, :, 0].reshape(-1)
    X['G'] = img[:, :, 1].reshape(-1)
    X['R'] = img[:, :, 2].reshape(-1)

    return X


def y_to_img(y):

    WIDTH = 500
    HEIGHT = 500

    y = y.reshape(HEIGHT, WIDTH)

    return y


def reshape_label(y):
    
    y = y.reshape(-1)

    return y

if __name__ == '__main__':

    print('Hello, home!')