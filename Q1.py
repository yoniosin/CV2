import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def getGaussPyramid(image, levels):
    gauss = {}
    for i in range(2, levels + 1):
        gauss[i] = cv.GaussianBlur(image, (0, 0), 2 ** i).astype(int)
    return gauss


def getLaplasPyramid(image, levels):
    g = getGaussPyramid(image, levels)

    laplas = {1: image.astype(int) - g[2], levels: g[levels]}
    for i in range(2, levels):
        laplas[i] = g[i] - g[i + 1]

    return laplas


def reconstructPyramid(pyramid_levels):
    img_size = pyramid_levels[1].shape
    result = np.zeros(img_size, dtype=int)
    for i in range(1, len(pyramid_levels) + 1):
        result += pyramid_levels[i]

    return result.astype(int)


if __name__ == '__main__':
    img = cv.imread('noy.jpg', 0)
    n = 6
    L = getLaplasPyramid(img, n)

    plt.subplot(1, 2, 1), plt.imshow(reconstructPyramid(L), cmap='gray'), plt.title('Reconstructed')
    plt.subplot(1, 2, 2), plt.imshow(img, cmap='gray'), plt.title('Original')
    plt.show()
