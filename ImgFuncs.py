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

    L = {}
    for i in range(1, levels + 1):
        if i == 1:
            L[i] = image.astype(int) - g[2]
        elif i == levels:
            L[i] = g[levels]
        else:
            L[i] = g[i] - g[i + 1]

    return L


def getRBGLaplacianPyramid(image, levels):
    RGBPyr = {}
    for i in range(1, 4):
        RGBPyr[i] = getLaplasPyramid(image[:, :, i-1], levels)

    return RGBPyr


def reconstructPyramid(pyramid_levels):
    img_size = pyramid_levels[1].shape
    result = np.zeros(img_size, dtype=int)
    for i in range(1, len(pyramid_levels) + 1):
        result += pyramid_levels[i]

    return result.astype(int)


def changeBackgroud(input_img, bg_mask, example_bg):

    return input_img * bg_mask + example_bg * np.logical_not(bg_mask)


 def calcGain(in_img, ex_img):
    n = 6;

    inPyr = getRBGLaplacianPyramid(in_img, n)
    exPyr = getRBGLaplacianPyramid(ex_img, n)

    energy_in = []
    energy_ex = []
    for rbg_idx in range(1, 4):
        for level_idx in range(1, n + 1):
            # TODO Im not sure if energy of each level supposed to be mat or scalar (sum of sum)
            energy_in.appand(cv.GaussianBlur(inPyr[rbg_idx][level_idx] ** 2, (0, 0), level_idx)) # TODO level_idx+1?
            energy_ex.appand(cv.GaussianBlur(exPyr[rbg_idx][level_idx] ** 2, (0, 0), level_idx)) # TODO level_idx+1?

