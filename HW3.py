import Q3
import matplotlib.pyplot as plt
from numpy.random import permutation
import cv2 as cv
import numpy as np
from collections import namedtuple


class Frame:
    Point = namedtuple('Point', ['x', 'y'])

    def __init__(self, idx, img):
        self.idx = idx
        self.img = img
        self.feature_points_vec = []
        self.cornerDetector()

    def GetIndexes(self, point, window_size):
        max_possible_size = min(point.x, self.img.shape[0] - point.x, point.y, self.img.shape[1] - point.y)
        if window_size > max_possible_size:
            raise ValueError

        x_idx = list(range(int(point.x - window_size / 2), int(point.x + window_size / 2 + 1)))
        y_idx = list(range(int(point.y - window_size / 2), int(point.y + window_size / 2 + 1)))

        return x_idx, y_idx

    def GetWindow(self, point, window_size):
            return self.GetIndexes(point, window_size)

    @staticmethod
    def CalculateSSD(a, b):
        return np.sum((a - b ** 2))

    @staticmethod
    def apllyAffineTrans(source_point, M):
        return np.dot(M[:, :2], source_point) + M[:, -1]

    def cornerDetector(self):
        gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)

        # find Harris corners
        gray = np.float32(gray)
        dst = cv.cornerHarris(gray, 2, 3, 0.04)
        dst = cv.dilate(dst, None)
        ret, dst = cv.threshold(dst, 0.01 * dst.max(), 255, 0)
        dst = np.uint8(dst)

        # find centroids
        ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)

        # define the criteria to stop and refine the corners
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

        for i in range(corners.shape[0]):
            self.feature_points_vec.append(self.Point(corners[i, 0], corners[i, 1]))

        # Now draw them
        # res = np.hstack((centroids, corners))
        # res = np.int0(res)

        # img[res[:,1],res[:,0]]=[0,0,255]
        # img[res[:, 3], res[:, 2]] = [0, 255, 0]
        # plt.figure()
        # plt.imshow(img)
        # plt.show()

        # return np.int0(corners)


# def calcAffineMatrix(reference_points, shifted_points):
#     reference_mat = np.zeros((6, 6))
#     res_vec = np.zeros(6)
#
#     for i in range(3):
#         reference_mat[2 * i, 0] = reference_points[i, 0]  # x_r
#         reference_mat[2 * i, 1] = reference_points[i, 1]  # y_r
#         reference_mat[2 * i, 4] = 1  # b1
#
#         reference_mat[2 * i + 1, 2] = reference_points[i, 0]  # x_r
#         reference_mat[2 * i + 1, 3] = reference_points[i, 1]  # y_r
#         reference_mat[2 * i + 1, 5] = 1  # b2
#
#         res_vec[2 * i] = shifted_points[i, 0]  # x
#         res_vec[2 * i + 1] = shifted_points[i, 1]  # y
#
#     affine_parameters_vec = np.linalg.solve(reference_mat, res_vec)
#
#     return affine_parameters_vec


def initChosenFeatures(corners):
    chosen_features = np.empty((6, 3, 2), dtype=np.float32)

    features_mat = [[1, 6, 13], [1, 9, 18], [1, 8, 14], [3, 10, 18], [1, 6, 13], [6, 13, 19]]

    for i in range(6):
        chosen_features[i, :, :] = corners[i][features_mat[i], :]

    return chosen_features


class SourceFrame(Frame):
    CoupledPoints = namedtuple('CoupledPoints', ['src_point', 'dst_point'])

    def __init__(self, src_img, dst_img_vec):
        super().__init__(0, src_img)
        self.frame_vec = [Frame(i, pic) for i, pic in enumerate(dst_img_vec)]
        self.frame_num = len(self.frame_vec)
        self.coupled_points = {i: [] for i in range(self.frame_num)}
        self.affinne_mat = {i: np.empty((2, 3)) for i in range(self.frame_num)}

    def calcAffineTrans(self, dst_frame_idx, selected_idx_vec):
        reference_pts = np.zeros((3, 2))
        shifted_pts = np.zeros((3, 2))
        for i, selected_idx in enumerate(selected_idx_vec):
            reference_pts[i, :] = self.coupled_points[dst_frame_idx][selected_idx].src_point
            shifted_pts[i, :] = self.coupled_points[dst_frame_idx][selected_idx].dst_point

        M = cv.getAffineTransform(reference_pts, shifted_pts)
        iM = np.zeros(M.shape)
        cv.invertAffineTransform(M, iM)
        return M, iM

    @staticmethod
    def SearchFeaturePoints(src_point, L, dst_point_vec):
        points_in_range = []
        for potential_point in dst_point_vec:
            if np.all(abs(np.asarray([potential_point.x - src_point.x, potential_point.y - src_point.y])) <= L / 2):
                points_in_range.append(potential_point)

        return points_in_range

    def FindBestPoint(self, src_point, dst_frame, potential_point_vec, window_size):
        src_window = self.GetWindow(src_point, window_size)
        ssd_vec = []
        for dst_point in potential_point_vec:
            try:
                ssd_vec.append(self.CalculateSSD(src_window, dst_frame.GetWindow(dst_point, window_size)))
            except ValueError:
                continue

        return potential_point_vec[np.argmin([ssd_vec])]

    def AutomaticMatch(self, dst_frame_idx, L, W):
        dst_frame = self.frame_vec[dst_frame_idx]
        for src_point in self.feature_points_vec:
            points_in_range = self.SearchFeaturePoints(src_point, L, dst_frame.feature_points_vec)
            if len(points_in_range) == 0:
                continue
            if len(points_in_range) == 1:
                best_point = points_in_range[0]
            else:
                try:
                    best_point = self.FindBestPoint(src_point, dst_frame, points_in_range, W)
                except ValueError:
                    continue

            self.AddCoupledPoints(dst_frame.idx, self.CoupledPoints(src_point, best_point))

    def AddCoupledPoints(self, dst_frame_idx, coupled_points):
        self.coupled_points[dst_frame_idx].append(coupled_points)

    def RANSAC(self, dst_frame_idx):
        inlier_group = []
        biggest_inlier = []
        best_M = np.empty((2, 3))
        for _ in range(100):
            rand_pts_idx = permutation(len(self.coupled_points[dst_frame_idx]))[:3]
            M, _ = self.calcAffineTrans(dst_frame_idx, rand_pts_idx)

            src_points_vec = self.coupled_points[dst_frame_idx].src_point
            dst_points_vec = self.coupled_points[dst_frame_idx].dst_point
            source_trans = [self.apllyAffineTrans(source_point, M) for source_point in src_points_vec]
            points_dist = calcEuclideanDist(source_trans, dst_points_vec)
            inlier_group.append(src_points_vec[points_dist < 10])

            if len(inlier_group) > len(biggest_inlier):
                biggest_inlier = inlier_group
                best_M = M

        self.affinne_mat[dst_frame_idx] = best_M


''' Aux Methods'''


def plotRconstImg(input, output):
    plt.subplot(121), plt.imshow(input), plt.title('Input')
    plt.subplot(122), plt.imshow(output), plt.title('Output')
    plt.show()


def calcEuclideanDist(estimated_points_list, real_points_list):
    dist_list = [np.linalg.norm(estimated_point, real_point) for estimated_point, real_point in
                 zip(estimated_points_list, real_points_list)]
    return np.asarray(dist_list)


# def reconstractImgs(chosen_features, list_of_frames):
#     reference_pts = chosen_features[0, :, :]
#
#     rows, cols, ch = list_of_frames[0].shape
#     M_list = []
#     iM_list = []
#     reconst_img = []
#
#     for i in range(1, len(list_of_frames)):
#         shifted_pts = chosen_features[i, :, :]
#         M, iM = self.calcAffineTrans(reference_pts, shifted_pts)
#         M_list.append(M)
#         iM_list.append(iM)
#         reconst_img.append(cv.warpAffine(frames[i], iM, (cols, rows)))
#
#     return reconst_img


# def ManualMatching():
#     corners = [cornerDetector(img) for img in frames]
#
#     chosen_features = initChosenFeatures(corners)
#     reconst_img_list = reconstractImgs(chosen_features, frames)
#
#     [plotRconstImg(frame, reconst) for frame, reconst in zip(frames, reconst_img_list)]


if __name__ == '__main__':
    frames_num = list(range(20, 100, 15))
    frames_names = ['extractedImgs/frame' + "%03d" % num + '.jpg' for num in frames_num]
    frames = [cv.imread(im) for im in frames_names]

    source_frame = SourceFrame(frames[0], frames[1:])
    for i in range(source_frame.frame_num):
        source_frame.AutomaticMatch(i, 50, 50)
    print('all done')
