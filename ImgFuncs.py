import cv2 as cv
import numpy as np


def normalize(image):
    return (255 * (image - image.min()) / (image.max() - image.min())).astype(np.uint8)


# //////////////////////////////// Q1 Funcs ////////////////////////
def getGaussPyramid(image, levels):
    g = {}
    for i in range(2, levels + 1):
        g[i] = cv.GaussianBlur(image, (0, 0), 2 ** i).astype(float)
    return g


def getLaplasPyramid(image, levels):
    g = getGaussPyramid(image, levels)

    L = {1: image.astype(float) - g[2], levels: g[levels]}
    for i in range(2, levels):
        L[i] = g[i] - g[i + 1]
    return L


def reconstructImgFromPyramid(pyramid_levels):
    img_size = pyramid_levels[1].shape
    result = np.zeros(img_size)
    for i in range(1, len(pyramid_levels) + 1):
        result += pyramid_levels[i]

    result = 255 * np.clip(result, 0, 1)
    return result.astype(np.uint8)


# ////////////////////////////// Q2 Funcs ////////////////////
def getRBGLaplacianPyramid(image, levels):
    RGBPyr = {}
    for i, color in enumerate(['R', 'G', 'B']):
        RGBPyr[color] = getLaplasPyramid(image[:, :, i], levels)

    return RGBPyr


def changeBackgroud(input_img, bg_mask, example_bg):
    return input_img * bg_mask + example_bg * np.logical_not(bg_mask)


def calcEnergy(RGBPyr, levels):
    energy_in = {}

    for color in ['R', 'G', 'B']:
        pyramid = RGBPyr[color]
        for j in range(1, levels + 1):
            key = color + str(j)
            energy_in[key] = cv.GaussianBlur(pyramid[j] ** 2, (0, 0), 2 ** (j + 1))

    return energy_in


def calcGain(ex_energy, in_energy, min_val, max_val, levels):
    gain = {}
    eps = 10 ** -4
    for color in ['R', 'G', 'B']:
        for j in range(1, levels + 1):
            key = color + str(j)
            gain[key] = np.clip(np.sqrt(ex_energy[key] / (in_energy[key] + eps)), min_val, max_val)

    return gain


def constructOutPyramid(gain, inRGBPyr, exRGBPyr, levels):
    outputPyr = {'R': {}, 'G': {}, 'B': {}}
    for color in ['R', 'G', 'B']:
        for j in range(1, levels + 1):
            key = color + str(j)
            if j == levels:
                outputPyr[color][j] = exRGBPyr[color][j]
            else:
                outputPyr[color][j] = (gain[key] * inRGBPyr[color][j])
    return outputPyr
