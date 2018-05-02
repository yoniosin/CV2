import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def getPyramid(image, levels):
    L = [np.empty(image.shape)] * levels
    for i in range(1, levels + 1):
        if i == 1:
            L[i - 1] = image - cv.GaussianBlur(image, (0, 0), 2 ** 2)
        elif i == n:
            L[i - 1] = cv.GaussianBlur(image, (0, 0), 2 ** n)
        else:
            L[i - 1] = cv.GaussianBlur(image, (0, 0), 2 ** i) - cv.GaussianBlur(image, (0, 0), 2 ** i + 1)

    return L


def reconstructPyramid(pyramid_levels):
    img_size = pyramid_levels[0].shape
    result = np.empty(img_size)
    for i in range(len(pyramid_levels)):
        result += pyramid_levels[i]

    normal = np.empty(pyramid_levels[0].shape)
    cv.normalize(result, normal, 0, 255, norm_type=cv.NORM_MINMAX)
    return normal


if __name__ == '__main__':
    n = 5
    L = getPyramid(cv.imread('noy.jpg'), n)
    for i in range(n):
        plt.imshow(L[i])
        plt.show()

    plt.imshow(reconstructPyramid(L))
    plt.show()
