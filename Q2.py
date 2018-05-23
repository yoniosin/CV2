import matplotlib.pyplot as plt
from ImgFuncs import *
import os
import re
from collections import namedtuple


def styleChange(input_data, style_data):
    in_img = cv.imread(input_data.image_path)[:, :, ::-1] / 255
    in_img_msk = cv.imread(input_data.mask_path)[:, :, ::-1]

    ex_img = (cv.imread(style_data.style_path)[:, :, ::-1]) / 255
    ex_bg = (cv.imread(style_data.mask_path)[:, :, ::-1]) / 255
    in_img_new_bg = changeBackgroud(in_img, in_img_msk, ex_bg)

    plt.subplot(1, 3, 1), plt.imshow(in_img), plt.title('Input Image')
    plt.subplot(1, 3, 2), plt.imshow(ex_img), plt.title('Example Image')
    plt.subplot(1, 3, 3), plt.imshow(in_img_new_bg), plt.title('New Bg Image')

    plt.show()
'''
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

    plt.subplot(1, 3, 1), plt.imshow(in_img), plt.title('Input Image')
    plt.subplot(1, 3, 2), plt.imshow(ex_img), plt.title('Example Image')
    plt.subplot(1, 3, 3), plt.imshow(output), plt.title('Image After Style Change')
    title = 'Transition from ' + input_data.name + ' to ' + style_data.name
    plt.suptitle(title)

    plt.savefig(title + '_fig')
    plt.show()
    plt.imsave(title, output)
'''


def parseInstructions(dictionary):
    input_path = './data/Inputs/imgs/'
    input_mask_path = './data/Inputs/masks/'
    inputData = namedtuple('inputData', ['name', 'image_path', 'mask_path', 'styles'])

    file_name_pattern = r'(.*)\.'  # extract name + extension

    in_dict = {}
    for file in os.listdir(input_path):
        pure_name = re.match(file_name_pattern, file).group(1)  # extract only the file's name (before '.')
        in_dict[file] = inputData(pure_name, input_path + file, input_mask_path + file, dictionary[file])

    example_path = './data/Examples/imgs/'
    example_mask_path = './data/Examples/bgs/'

    styleData = namedtuple('styleData', ['name', 'style_path', 'mask_path'])
    ex_dict = {}

    for file in os.listdir(example_path):
        pure_name = re.match(file_name_pattern, file).group(1)  # extract only the file's name (before '.')
        ex_dict[file] = styleData(pure_name, example_path + file, example_mask_path + pure_name + '.jpg')

    return in_dict, ex_dict


def styleChangeWrapper(input_dict, example_dict):
    # receive dictionaries containing all needed paths, and call styleChange for every requested combination
    for image_key in input_dict:
        image = input_dict[image_key]
        for style_key in image.styles:
            style = example_dict[style_key]
            styleChange(image, style)


if __name__ == '__main__':
    instructions_dict = {'0004_6.png': ['16.png', '21.png'], '0006_001.png': ['0.png', '9.png', '10.png']}
    inputs, examples = parseInstructions(instructions_dict)
    styleChangeWrapper(inputs, examples)

    print("All done :)")
