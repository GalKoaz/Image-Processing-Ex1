# Image-Processing Exercise 1


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Content</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#gamma-correction">Gamma Correction</a></li>
    <li><a href="#code-details">Code Details</a></li>
    <li><a href="#languages-and-tools">Languages and Tools</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

----------------

<!-- ABOUT THE PROJECT -->
# About The Project
**_Image-Processing Exercise 1:_**

The main purpose of this exercise is to get you acquainted with Python’s basic syntax and some of its
image processing facilities. This exercise covers:

* Loading grayscale and RGB image representations.
* Displaying figures and images.
* Transforming RGB color images back and forth from the YIQ color space.
* Performing intensity transformations: histogram equalization.
* Performing optimal quantization


``` Version Python 3.9```

## Gamma Correction

<img src="https://i.ibb.co/0r7RKML/2.png" alt="2" border="0">

<img src="https://i.ibb.co/R2ch5D4/1.png" alt="1" border="0">

<img src="https://i.ibb.co/pzr4dTk/3.png" alt="3" border="0">


---------------------

## Code Details

Displaying an image - function that utilizes imReadAndConvert to display a given image file in a given representation.

The function should have the following interface:

```python
def imDisplay(filename:str, representation:int)->None:
    """
    Reads an image as RGB or GRAY_SCALE and displays it`
    :param filename: The path to the image
    :param representation: grayscale(1) or RGB(2)
    :return: None
    """
```

Transforming an RGB image to YIQ color space - Write two functions that transform an RGB image into the YIQ color space (mentioned in the lecture)
and vice versa. Given the red (R), green (G), and blue (B) pixel components of an RGB color image,
the corresponding luminance (Y), and the chromaticity components (I and Q) in the YIQ color space are
linearly related as follows:

<a href="https://imgbb.com/"><img src="https://i.ibb.co/JcfrGkh/Untitled.png" alt="Untitled" border="0"></a>

The two functions should have the following interfaces:
```python
def transformRGB2YIQ(imRGB:np.ndarray)->np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
```

```python
def transformYIQ2RGB(imYIQ:np.ndarray)->np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
```

Histogram equalization - Write a function that performs histogram equalization of a given grayscale or RGB image. The function
should also display the input and the equalized output image. 

The function should have the following interface:

```python
def histogramEqualize(imOrig:np.ndarray)->(np.ndarray,np.ndarray,np.ndarray):
    """
    Equalizes the histogram of an image
    :param imgOrig: Original image
    :return: (imgEq,histOrg,histEQ)
    """
```

Optimal image quantization - Write a function that performs optimal quantization of a given grayscale or RGB image. The function
should return:

* A list of the quantized image in each iteration
* A list of the MSE error in each iteration

The function should have the following interface:

```python
def quantizeImage(imOrig:np.ndarray, nQuant:int, nIter:int)->(List[np.ndarray],List[float]):
    """
    Quantized an image in to **nQuant** colors
    :param imOrig: The original image (RGB or Gray scale)
    :param nQuant: Number of colors to quantize the image to
    :param nIter: Number of optimization loops
    :return: (List[qImage_i],List[error_i])
    """
```

Gamma Correction - function that performs gamma correction on an image with a given γ.
For this task, you’ll be using the OpenCV functions createTrackbar to create the slider and display
it, since it’s OpenCV’s functions, the image will have to be represented as BGR.

```python
def gammaDisplay(img_path:str, rep:int)->None:
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
```

---------------------

## Languages and Tools

  <div align="center">
  
 <code><img height="40" width="40" src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png"></code> 
 <code><img height="40" width="80" src="https://matplotlib.org/_static/logo2_compressed.svg"/></code>
 <code><img height="40" width="40" src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/PyCharm_Icon.svg/1024px-PyCharm_Icon.svg.png"/></code>
 <code><img height="40" height="40" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/git/git.png"></code>
 <code><img height="40" height="40" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/terminal/terminal.png"></code>
  </div>


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [Python](https://www.python.org/)
* [Matplotlib](https://matplotlib.org/)
* [Git](https://git-scm.com/)
* [Pycharm](https://www.jetbrains.com/pycharm/)
* [Git-scm](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)


<!-- CONTACT -->
## Contact


 Gal - [here](https://github.com/GalKoaz/)


Project Link: [here](https://github.com/GalKoaz/Image-Processing-Ex1)

___

Copyright © _This Project was created on March 24, 2022, by [Gal](https://github.com/GalKoaz/)_.
