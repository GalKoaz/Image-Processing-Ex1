"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2
YIQ2RGB = np.array([[0.299, 0.587, 0.114],
                    [0.59590059, -0.27455667, -0.32134392],
                    [0.21153661, -0.52273617, 0.31119955]])


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 206260168


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    Normalization of data is transforming the data to appear on the same scale across all the records.
    You can normalize data between 0 and 1 range by using the formula (data – np. min(data)) / (np. max(data) – np.min(data)).
    """
    image = cv2.imread(filename)
    if representation == LOAD_RGB:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif representation == LOAD_GRAY_SCALE:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return (image - np.min(image)) / (np.max(image) - np.min(image))


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    image = imReadAndConvert(filename, representation)
    plt.imshow(image)
    if representation == LOAD_GRAY_SCALE:
        plt.gray()
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    if len(imgRGB.shape) == 3:
        return imgRGB.dot(YIQ2RGB.T)


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    if len(imgYIQ.shape) == 3:
        return imgYIQ.dot(YIQ2RGB.T)


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    pass


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    pass
