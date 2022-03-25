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
YIQ2RGB = np.array([[0.299, 0.587, 0.114],
                    [0.59590059, -0.27455667, -0.32134392],
                    [0.21153661, -0.52273617, 0.31119955]])


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    # identification number
    return 206260168


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
    # Read and Convert the image and check if the image is load gray we plot with gray scale
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
    # .Dot product of two arrays.
    # .T The transposed array, Same as self.transpose().
    if imgRGB.shape == 3:
        return imgRGB.dot(YIQ2RGB.T)


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    # .Dot product of two arrays.
    # .T The transposed array, Same as self.transpose().
    if imgYIQ.shape == 3:
        return imgYIQ.dot(YIQ2RGB.T)


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    image = transformRGB2YIQ(cv2.normalize(imgOrig.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX))
    if len(imgOrig.shape) == 3:
        img_eq = cv2.normalize(image[:, :, 0], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    else:
        img_eq = (imgOrig * 255).astype('uint8')

    hist_org, bins = np.histogram(img_eq.flatten(), 256, [0, 256])

    # Instillation of Plot 1 with the settings: title, the plot and the histogram

    # figure, (plot1, plot2) = plt.subplots(1, 2, figsize=(12, 7))
    # plot1.set_title('Original Image Histogram & CDF')
    # plot1.plot((hist_org.cumsum() * hist_org.max()) / hist_org.cumsum().max(), color='blue')
    # plot1.hist(img_eq.flatten(), 256, [0, 256], color='red')
    # plot1.legend(('CDF', 'Histogram'), loc='best')

    # Equalized Image with linear CDF Masked arrays are arrays that may have missing or invalid entries.
    # The numpy.ma module provides a nearly work-alike replacement for numpy that supports data arrays with masks.
    cdf_m = np.ma.masked_equal(hist_org.cumsum(), 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    linear_img = np.ma.filled(cdf_m, 0).astype('uint8')[img_eq]
    hist_eq, bins = np.histogram(linear_img.flatten(), 256, [0, 255])

    # plot2.set_title('Post-Equalization Image Histogram & CDF ')
    # plot2.plot(hist_eq.cumsum() * hist_eq.max() / hist_eq.cumsum().max(), color='blue')
    # plot2.hist(linear_img.flatten(), 256, [0, 255], color='red')
    # plot2.legend(('CDF', 'Histogram'), loc='best')
    # plt.xlim([0, 256])
    # plt.show()

    # Needing to Transform equalized image back to RGB If the image is 8-bit unsigned, it is displayed as is. If the
    # image is 16-bit unsigned or 32-bit integer, the pixels are divided by 256. That is, the value range [0,
    # 255*256] is mapped to [0,255]. If the image is 32-bit floating-point, the pixel values are multiplied by 255.
    # That is, the value range [0,1] is mapped to [0,255].
    if len(imgOrig.shape) == 3:
        image[:, :, 0] = cv2.normalize(img_eq.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        img_eq = (transformYIQ2RGB(image) * 255).astype('uint8')
    img_eq = cv2.LUT(img_eq, np.ma.filled(cdf_m, 0).astype('uint8'))
    return img_eq, hist_org, hist_eq


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    pass

