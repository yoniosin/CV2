import matplotlib.pyplot as plt
from ImgFuncs import *


def styleChange(input_path, input_bg_path, example_path, example_bg_path):
    in_img = cv.imread(input_path)[:, :, ::-1] / 255
    in_img_msk = cv.imread(input_bg_path)[:, :, ::-1] > 100

    ex_img = (cv.imread(example_path)[:, :, ::-1]) / 255
    ex_bg = (cv.imread(example_bg_path)[:, :, ::-1]) / 255
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
    for k, color in enumerate(['R', 'G', 'B']):
        output[:, :, k] = reconstructImgFromPyramid(outputPyr[color])
    plt.show()

    plt.imshow(output)
    plt.show()


if __name__ == '__main__':
    input_name = ['0004_6.png', '0006_001.png']
    input_image = ['./data/Inputs/imgs/' + s for s in input_name]
    input_mask = ['./data/Inputs/masks/' + s for s in input_name]

    example_dict = {0: ['16', '21'], 1: ['0', '9', '10']}
    for i, _ in enumerate(input_name):
        example_name = example_dict[i]
        example_image = ['./data/Examples/imgs/' + s + '.png' for s in example_name]
        example_bg = ['./data/Examples/bgs/' + s + '.jpg' for s in example_name]
        for j, _ in enumerate(example_name):
            styleChange(input_image[i], input_mask[i], example_image[j], example_bg[j])

    in_img = cv.imread('data/Inputs/imgs/0004_6.png')[:, :, ::-1] / 255
    in_img_msk = cv.imread('data/Inputs/masks/0004_6.png')[:, :, ::-1] > 100

    ex_img = (cv.imread('data/Examples/imgs/6.png')[:, :, ::-1]) / 255
    ex_bg = (cv.imread('data/Examples/bgs/6.jpg')[:, :, ::-1]) / 255
    plt.imshow(ex_bg)
    print("All done :)")
