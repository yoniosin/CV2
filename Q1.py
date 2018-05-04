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

<<<<<<< Updated upstream
    return result.astype(np.uint)
=======
    # normal = np.zeros(pyramid_levels[0].shape)
    # cv.normalize(result, normal, 0, 255, norm_type=cv.NORM_MINMAX)
    # return normal.astype(np.uint)
    return result.astype(np.int)
>>>>>>> Stashed changes


if __name__ == '__main__':
    plt.imshow(cv.imread('noy.jpg', 0), cmap='gray')
    plt.show()
    n = 3
    L = getPyramid(cv.imread('noy.jpg', 0), n)
    normal = np.zeros(L[0].shape)
    for i in range(n):
<<<<<<< Updated upstream
        plt.imshow(L[i].astype(np.uint))
=======
        cv.normalize(L[i], normal, 0, 255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        plt.imshow(normal, cmap='gray')
>>>>>>> Stashed changes
        plt.show()

    plt.imshow(reconstructPyramid(L), cmap='gray')
    plt.show()
