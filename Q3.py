import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from numpy.random import permutation

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

    # img[res[:,1],res[:,0]]=[0,0,255]
    img[res[:,3],res[:,2]] = [0,255, 0]
    plt.figure()
    plt.imshow(img)
    # plt.show()

    return np.int0(corners)

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

def calcAffineMatrix(reference_points, shifted_points):
    reference_mat = np.zeros((6, 6))
    res_vec = np.zeros(6)

    for i in range(3):
        reference_mat[2 * i, 0] = reference_points[i, 0] #x_r
        reference_mat[2 * i, 1] = reference_points[i, 1] #y_r
        reference_mat[2 * i, 4] = 1 #b1

        reference_mat[2 * i + 1, 2] = reference_points[i, 0]  # x_r
        reference_mat[2 * i + 1, 3] = reference_points[i, 1]  # y_r
        reference_mat[2 * i + 1, 5] = 1  # b2

        res_vec[2 * i] = shifted_points[i, 0] #x
        res_vec[2 * i + 1] = shifted_points[i, 1] #y

    affine_parameters_vec = np.linalg.solve(reference_mat, res_vec)

    return affine_parameters_vec


def initChosenFeatures(corners):
    chosen_features = np.empty((6, 3, 2), dtype=np.float32)

    features_mat = [[1, 6, 13],[1, 9, 18],[1, 8, 14],[3, 10, 18],[1, 6, 13],[6, 13, 19]]

    for i in range(6):
        chosen_features[i, :, :] = corners[i][features_mat[i], :]

    return chosen_features


def calcAffineTrans(reference_pts, shifted_pts):
    M = cv.getAffineTransform(reference_pts, shifted_pts)
    iM = np.zeros(M.shape)
    cv.invertAffineTransform(M, iM)
    return M, iM


def reconstractImgs(corners_features, list_of_frames):
    reference_pts = chosen_features[0, :, :]

    rows, cols, ch = list_of_frames[0].shape
    M_list = []
    iM_list = []
    reconst_img = []

    for i in range(1, len(list_of_frames)):
        shifted_pts = chosen_features[i, :, :]
        M, iM = calcAffineTrans(reference_pts, shifted_pts)
        M_list.append(M)
        iM_list.append(iM)
        reconst_img.append(cv.warpAffine(frames[i], iM, (cols, rows)))

    return reconst_img


def plotRconstImg(input, output):
    plt.subplot(121), plt.imshow(input), plt.title('Input')
    plt.subplot(122), plt.imshow(output), plt.title('Output')
    plt.show()


def RANSAC(source_list_points, dst_list_poitns):
    rand_pts_idx = permutation(len(source_list_points))[:3]
    M, iM = calcAffineTrans(source_list_points[rand_pts_idx], source_list_points[rand_pts_idx])


if __name__ == '__main__':
    # video_path = 'pasta.mp4'
    # extractImages(video_path)
    # makeVideoMask('extractedImgs')
    # createVideo('extractedImgs', 'masks', 'videoSeg.avi', 30)
    frames_num = list(range(20, 100, 15))
    frames_names = ['extractedImgs/frame' + "%03d" % num + '.jpg' for num in frames_num]
    frames = [cv.imread(im) for im in frames_names]
    corners = [cornerDetector(img) for img in frames]

    chosen_features = initChosenFeatures(corners)
    reconst_img_list = reconstractImgs(chosen_features, frames)

    [plotRconstImg(frame, reconst) for frame, reconst in zip(frames, reconst_img_list)]



    # [mathc(frames[0], frame) for frame in frames]
    print('all done')