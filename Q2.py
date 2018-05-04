import matplotlib.pyplot as plt
from ImgFuncs import *

if __name__ == '__main__':
    in_img = cv.imread('data/Inputs/imgs/0004_6.png')[:, :, ::-1]/255
    in_img_msk = cv.imread('data/Inputs/masks/0004_6.png')[:, :, ::-1]/255

    ex_img = cv.imread('data/Examples/imgs/6.png')[:, :, ::-1]/255
    ex_bg = cv.imread('data/Examples/bgs/6.jpg')[:, :, ::-1] / 255

    in_img_new_bg = changeBackgroud(in_img, in_img_msk, ex_bg)

    plt.subplot(1, 3, 1), plt.imshow(in_img), plt.title('Input Image')
    plt.subplot(1, 3, 2), plt.imshow(ex_img), plt.title('Example Image')
    plt.subplot(1, 3, 3), plt.imshow(new_bg), plt.title('New Bg Image')

    plt.show()

    n = 6
    input_pyr = getLaplasPyramid(in_img_new_bg, n)
    example_pyr = getLaplasPyramid(ex_img, n)


