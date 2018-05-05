import matplotlib.pyplot as plt
from ImgFuncs import *

if __name__ == '__main__':
    in_img = cv.imread('data/Inputs/imgs/0004_6.png')[:, :, ::-1] / 255
    in_img_msk = cv.imread('data/Inputs/masks/0004_6.png')[:, :, ::-1] > 100

    ex_img = (cv.imread('data/Examples/imgs/6.png')[:, :, ::-1]) / 255
    ex_bg = (cv.imread('data/Examples/bgs/6.jpg')[:, :, ::-1]) / 255
    plt.imshow(ex_bg)

    in_img_new_bg = changeBackgroud(in_img, in_img_msk, ex_bg)

    plt.subplot(1, 3, 1), plt.imshow(in_img), plt.title('Input Image')
    plt.subplot(1, 3, 2), plt.imshow(ex_img), plt.title('Example Image')
    plt.subplot(1, 3, 3), plt.imshow(in_img_new_bg), plt.title('New Bg Image')

    plt.show()

    n = 6
    inRGBPyr = getRBGLaplacianPyramid(in_img_new_bg, n)
    examplePyr = getRBGLaplacianPyramid(ex_img, n)

    input_energy_pyr = calcEnergy(inRGBPyr, n)
    ex_energy_pyr = calcEnergy(examplePyr, n)

    gain = calcGain(ex_energy_pyr, input_energy_pyr, 0.9, 2.8, n)
    outputPyr = constructOutPyramid(gain, inRGBPyr, examplePyr, n)

    output = np.empty(in_img.shape, dtype=np.uint8)
    for i, color in enumerate(['R', 'G', 'B']):
        output[:, :, i] = reconstructImgFromPyramid(outputPyr[color])
    plt.show()

    plt.imshow(output)
    plt.show()
    print("All done :)")
