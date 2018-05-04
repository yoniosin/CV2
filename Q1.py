import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def getPyramid(image, levels):
    L = [np.empty(image.shape)] * levels
    for i in range(1, levels + 1):
        if i == 1:
            L[i - 1] = image.astype(int) - cv.GaussianBlur(image, (0, 0), 2 ** 2)
        elif i == n:
            L[i - 1] = cv.GaussianBlur(image, (0, 0), 2 ** n).astype(int)
        else:
            L[i - 1] = cv.GaussianBlur(image, (0, 0), 2 ** i).astype(int) - cv.GaussianBlur(image, (0, 0), 2 ** i + 1)

    return L


def reconstructPyramid(pyramid_levels):
    img_size = pyramid_levels[0].shape
    result = np.zeros(img_size)
    for i in range(len(pyramid_levels)):
        result += pyramid_levels[i]

    return result.astype(np.int)


if __name__ == '__main__':
    img = cv.imread('noy.jpg', 0)
    n = 6
    L = getPyramid(img, n)
    normal = np.zeros(L[0].shape)

    plt.subplot(1,2,1), plt.imshow(reconstructPyramid(L), cmap='gray')
    plt.subplot(1,2,2), plt.imshow(img, cmap='gray')
    plt.show()
