import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import nms


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


def createVideo(image_folder, mask_folder, video_name, fps):

    images = os.listdir(image_folder)
    masks = os.listdir(mask_folder)

    frame = cv.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    video = cv.VideoWriter(video_name, fourcc, fps, (width, height))

    for i, (image, mask) in enumerate(zip(images, masks)):
        curr_im = cv.imread(image_folder + '/' + image)
        curr_mask = cv.imread(mask_folder + '/' + mask) / 255
        proc_im = np.uint8(curr_im * curr_mask)
        cv.imwrite("vid/vid%03d.jpg" % i, proc_im)  # save frame as JPEG file
        video.write(proc_im)

    cv.destroyAllWindows()
    video.release()


def cornerDetector(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    # find Harris corners
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray,2,3,0.04)
    dst = cv.dilate(dst,None)
    ret, dst = cv.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

    # Now draw them
    res = np.hstack((centroids,corners))
    res = np.int0(res)
    img[res[:,1],res[:,0]]=[0,0,255]
    img[res[:,3],res[:,2]] = [0,255,0]

    cv.imshow('dst', img)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()

    plt.show()

def mathc(img1,img2):
    # Initiate ORB detector
    orb = cv.ORB_create()

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x : x.distance)

    img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:6] ,None, flags=2)

    plt.imshow(img3),plt.show()


if __name__ == '__main__':
    video_path = 'glass.mp4'
    extractImages(video_path)
    # makeVideoMask('pumpkinImgs')
    # createVideo('pumpkinImgs', 'masks', 'videoSeg.avi', 30)
    frames_num = list(range(0, 40, 6))
    frames_names = ['extractedImgs/frame' + "%03d" % num + '.jpg' for num in frames_num]
    frames = [cv.imread(im) for im in frames_names]
    [cornerDetector(img) for img in frames]
    # [mathc(frames[0], frame) for frame in frames]
    print('all done')