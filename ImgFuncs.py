import cv2 as cv
import numpy as np


def normalize(image):
    return (255 * (image - image.min()) / (image.max() - image.min())).astype(np.uint)


def getGaussPyramid(image, levels):
    g = {}
    for i in range(2, levels + 1):
        g[i] = cv.GaussianBlur(image, (0, 0), 2 ** i).astype(int)
    return g


def getLaplasPyramid(image, levels):
    g = getGaussPyramid(image, levels)

    L = {1: image.astype(int) - g[2], levels: g[levels]}
    for i in range(2, levels):
        L[i] = g[i] - g[i + 1]

    return L


def getRBGLaplacianPyramid(image, levels):
    RGBPyr = {}
    for i, color in enumerate(['R', 'G', 'B']):
        RGBPyr[color] = getLaplasPyramid(image[:, :, i - 1], levels)

    return RGBPyr


def reconstructPyramid(pyramid_levels):
    img_size = pyramid_levels[1].shape
    result = np.zeros(img_size, dtype=int)
    for i in range(1, len(pyramid_levels) + 1):
        result += pyramid_levels[i]

    return result.astype(int)


def changeBackgroud(input_img, bg_mask, example_bg):
    return input_img * bg_mask + example_bg * np.logical_not(bg_mask)


def calcEnergy(in_img, levels):
    inPyr = getRBGLaplacianPyramid(in_img, levels)
    energy_in = {}

    for i in ['R', 'G', 'B']:
        pyramid = inPyr[i]
        for j in range(1, levels + 1):
            key = i + str(j)
            energy_in[key] = cv.GaussianBlur(pyramid[j] ** 2, (0, 0), 2 ** (j + 1))
