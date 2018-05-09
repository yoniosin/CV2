import matplotlib.pyplot as plt

from ImgFuncs import *

if __name__ == '__main__':
    img = cv.imread('noy.jpg', 0) / 255
    n = 6
    L = getLaplasPyramid(img, n)

    for i in range(1, L.__len__()+1):
        plt.imshow(normalize(L[i]), cmap='gray')
        plt.show()

    recon = reconstructImgFromPyramid(L)
    diffImg = img - recon
    plt.subplot(1, 3, 1), plt.imshow(recon, cmap='gray'), plt.title('Reconstructed')
    plt.subplot(1, 3, 2), plt.imshow(img, cmap='gray'), plt.title('Original')
    plt.subplot(1, 3, 2), plt.imshow(diffImg, cmap='gray'), plt.title('Difference ' + str(np.linalg.norm(diffImg)))
    plt.show()
