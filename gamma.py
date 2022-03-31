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

import numpy as np

from ex1_utils import LOAD_GRAY_SCALE, LOAD_RGB
import cv2


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    if rep == LOAD_GRAY_SCALE:
        image = cv2.imread(img_path, 0) / 255
    else:
        image = cv2.imread(img_path) / 255

    def tracker(value):
        cv2.imshow('Gamma Correction', np.power(image, value / 100))

    cv2.namedWindow('Gamma Correction')
    cv2.createTrackbar('Gamma', 'Gamma Correction', 100, 200, tracker)
    tracker(100)
    cv2.waitKey()


def main():
    gammaDisplay('testImg1.jpg', LOAD_RGB)


if __name__ == '__main__':
    main()
