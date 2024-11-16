import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2


def load_image(path):
    """
    Loads an image from the specified file path.

    Parameters:
        path (str): Path to the image file.

    Returns:
        np.ndarray: The loaded image as a NumPy array.
    """

    return cv2.imread(path)


def load_label(path):
    """
    Loads a label file from the specified path.

    Parameters:
        path (str): Path to the label file.

    Returns:
        np.ndarray: The loaded label data as a NumPy array.
    """

    return np.loadtxt(path, dtype=np.uint8)


def save_image(img, output_path):
    """
    Saves an image to the specified output path.

    Parameters:
        img (np.ndarray): The image to be saved.
        output_path (str): Path where the image will be saved.

    Returns:
        None
    """

    plt.imsave(output_path, img)

    return None


def resize_img(img):
    """
    Resizes an image to a fixed size of 500x500 pixels.

    Parameters:
        img (np.ndarray): The input image to resize.

    Returns:
        np.ndarray: The resized image.
    """

    WIDTH = 500
    HEIGHT = 500

    img = cv2.resize(img, (WIDTH, HEIGHT))

    return img


def img_to_X(img):
    """
    Converts an image into a DataFrame with separate columns for each color channel.

    Parameters:
        img (np.ndarray): The input image as a NumPy array.

    Returns:
        pd.DataFrame: DataFrame with columns 'B', 'G', and 'R' containing pixel values.
    """

    X = pd.DataFrame()

    X['B'] = img[:, :, 0].reshape(-1)
    X['G'] = img[:, :, 1].reshape(-1)
    X['R'] = img[:, :, 2].reshape(-1)

    return X


def y_to_img(y):
    """
    Reshapes a 1D array of pixel labels into a 2D image with fixed dimensions.

    Parameters:
        y (np.ndarray): The input 1D array of pixel labels.

    Returns:
        np.ndarray: A 2D image with the dimensions 500x500.
    """

    WIDTH = 500
    HEIGHT = 500

    y = y.reshape(HEIGHT, WIDTH)

    return y


def reshape_label(y):
    """
    Reshapes a 2D array of labels into a 1D array.

    Parameters:
        y (np.ndarray): The input 2D array of labels.

    Returns:
        np.ndarray: A 1D array of labels.
    """
    
    y = y.reshape(-1)

    return y


if __name__ == '__main__':

    print('Hello, home!')