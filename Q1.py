import matplotlib.pyplot as plt
from ImgFuncs import *

if __name__ == '__main__':
    img = cv.imread('noy.jpg', 0)
    n = 6
    L = getLaplasPyramid(img, n)

    for i in range(1, L.__len__()+1):
        plt.imshow(normalize(L[i]), cmap='gray')
        plt.show()

    plt.subplot(1, 2, 1), plt.imshow(reconstructPyramid(L), cmap='gray'), plt.title('Reconstructed')
    plt.subplot(1, 2, 2), plt.imshow(img, cmap='gray'), plt.title('Original')
    plt.show()
