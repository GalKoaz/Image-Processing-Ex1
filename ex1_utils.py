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

# ----------------------------------------------------------------------------
# Created By : Gal Koaz
# Created Date : 24-03-2022
# Python version : '3.8'
# ---------------------------------------------------------------------------

from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2
YIQ2RGB = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 206260168  # identification number


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested :param filename: The path to the image :param
    representation: GRAY_SCALE or RGB :return: The image object
    """
    # Normalization of data is transforming the data to
    # appear on the same scale across all the records. You can normalize data between 0 and 1 range by using the
    # formula (data – np. min(data)) / (np. max(data) – np.min(data)).
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
    return np.dot(imgRGB, YIQ2RGB.T)  # .T The transposed array, Same as self.transpose().


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    return np.dot(imgYIQ, np.linalg.inv(YIQ2RGB).T)  # .T The transposed array, Same as self.transpose().


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    img = np.copy(imgOrig)
    if img.ndim == 3:
        img_yiq = transformRGB2YIQ(imgOrig)  # Convert to YIQ
        imgOrig = np.copy(img_yiq[:, :, 0])  # We Saving the  Y channel
    else:  # Case grayscale
        img_yiq = imgOrig
    imgOrig = imgOrig * 255
    imgOrig = (np.around(imgOrig)).astype('uint8')  # Rounding the values, cast the matrix to integers
    hist, bins = np.histogram(imgOrig.flatten(), 256, [0, 255])  # Creating the histogram of the original image
    cumsum = hist.cumsum()  # calculate "Cumsum"
    img_scale = np.ma.masked_equal(cumsum, 0)
    img_scale = (img_scale - img_scale.min()) * 255 / (img_scale.max() - img_scale.min())  # Scaling to our histogram
    after_scale = np.ma.filled(img_scale, 0).astype('uint8')  # cast the matrix values to integers

    imgEq = after_scale[imgOrig.astype('uint8')]  # mapping every point in "Cumsum" to new point
    histEQ, bins2 = np.histogram(imgEq.flatten(), 256, [0, 256])  # Creating the histogram of the new image

    if img.ndim == 3:  # Checking if the image is RGB
        img_yiq[:, :, 0] = imgEq / 255
        imgEq = transformYIQ2RGB(img_yiq)  # Convert back to RGB

    return imgEq, hist, histEQ


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    img = np.copy(imOrig)
    if img.ndim == 3:
        imgYIQ = transformRGB2YIQ(imOrig)  # Convert to YIQ
        imOrig = np.copy(imgYIQ[:, :, 0])  # We Saving the  Y channel
    else:  # Case grayscale
        imgYIQ = imOrig

    imOrig = cv2.normalize(imOrig, None, 0, 255, cv2.NORM_MINMAX)  # Normalize to [0, 255]
    imOrig = imOrig.astype('uint8')  # Casting the matrix to integers
    hist, bins = np.histogram(imOrig, 256, [0, 255])  # Calculate a histogram of the original image

    # Find The boundaries
    z = np.zeros(nQuant + 1, dtype=int)  # z is an array that will represents the boundaries
    for i in range(1, nQuant):
        z[i] = z[i - 1] + int(255 / nQuant)  # Divide the intervals
    z[nQuant] = 255  # The left border will always start at 0 and the right border will always end at 255
    q = np.zeros(nQuant)  # q is an array that represent the values of the boundaries

    images_list = list()  # Creating image return list
    mse_list = list()  # Creating mse errors return list

    for i in range(nIter):
        img_new = np.zeros(imOrig.shape)  # Initialize a matrix with 0 in the original image size
        for j in range(len(q)):  # Every j is a cell
            if j == len(q) - 1:  # The last iterate of j
                right = z[j + 1] + 1
            else:
                right = z[j + 1]
            range_cell = np.arange(z[j], right)
            q[j] = np.average(range_cell, weights=hist[z[j]:right])
            mat = np.logical_and(imOrig >= z[j], imOrig < right)  # Matrix that is initialized in T/F
            img_new[mat] = q[j]  # Where there is a T in the matrix we will update the new value
        mse_list.append(np.sum(np.square(np.subtract(img_new, imOrig))) / imOrig.size)  # According to mse formula

        if img.ndim == 3:
            imgYIQ[:, :, 0] = img_new / 255
            img_new = transformYIQ2RGB(imgYIQ)  # Convert back to RGB
        images_list.append(img_new)  # we appending the image to the images list

        for boundary in range(1, len(z) - 1):  # Each boundary become to be a middle of 2 means
            z[boundary] = (q[boundary - 1] + q[boundary]) / 2

        if len(mse_list) >= 2:
            if np.abs(mse_list[-1] - mse_list[-2]) <= 0.000001:
                break

    return images_list, mse_list
