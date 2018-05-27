import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def extractImages(video_path):
    vidcap = cv.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
        cv.imwrite("extractedImgs/frame%03d.jpg" % count, image)     # save frame as JPEG file
        success, image = vidcap.read()
        count += 1


def makeImMask(im):
    Z = im.reshape((-1, 3))

    clt = KMeans(2)
    clt.fit(Z)

    label_pix = np.asanyarray([sum(clt.labels_ == 0), sum(clt.labels_ == 1)])
    obj_label = np.argmin(label_pix)
    mask = np.zeros(clt.labels_.shape)
    mask[np.where(clt.labels_ == obj_label)] = 1
    mask = np.uint8(mask.reshape((im.shape[0], im.shape[1])) * 255)

    return mask


def makeVideoMask(extract_imgs_path):
    for i, filename in enumerate(os.listdir(extract_imgs_path)):
        mask = makeImMask(cv.imread(extract_imgs_path + '/' + filename))
        cv.imwrite("masks/mask%03d.jpg" % i, mask)  # save frame as JPEG file


def createVideo(image_folder, mask_folder, video_name):

    images = os.listdir(image_folder)
    masks = os.listdir(mask_folder)

    frame = cv.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    video = cv.VideoWriter(video_name, fourcc, 30, (width, height))

    for i, (image, mask) in enumerate(zip(images, masks)):
        curr_im = cv.imread(image_folder + '/' + image)
        curr_mask = cv.imread(mask_folder + '/' + mask) / 255
        proc_im = np.uint8(curr_im * curr_mask)
        cv.imwrite("vid/vid%03d.jpg" % i, proc_im)  # save frame as JPEG file
        video.write(proc_im)

    cv.destroyAllWindows()
    video.release()


if __name__ == '__main__':
    video_path = 'chair.mp4'
    extractImages(video_path)
    makeVideoMask('extractedImgs')
    createVideo('extractedImgs', 'masks', 'videoSeg.avi')

    print('all dona')