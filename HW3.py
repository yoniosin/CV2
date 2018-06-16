import matplotlib.pyplot as plt
from numpy.random import permutation
from Q3 import *
from klt import *
from sklearn.utils.extmath import randomized_svd as svd_t

class Frame:
    Point = namedtuple('Point', ['x', 'y'])

    def __init__(self, idx, img, reference_img):
        self.idx = idx
        self.img = img
        self.feature_points_vec = []
        self.reference_img = reference_img
        self.cornerDetector()

        self.img_with_harris = self.img
        # for point in self.feature_points_vec:
        #     self.ZeroPixelsInWindow(point, 10, self.img_with_harris, 255)

    def ZeroPixelsInWindow(self, center, window_size, img, color):
        try:
            x_idx_vec, y_idx_vec = self.GetIndexes(center, window_size)
        except ValueError:
            return
        for x_idx in x_idx_vec:
            for y_idx in y_idx_vec:
                img[y_idx, x_idx] = [0, 0, color]

    @staticmethod
    def ThrowError():
        raise FrameError

    def GetIndexes(self, point, window_size):
        max_possible_size = min(point.x, self.img.shape[1] - point.x, point.y, self.img.shape[0] - point.y)
        if window_size > max_possible_size:
            self.ThrowError()

        x_idx = list(range(int(point.x - window_size / 2), int(point.x + window_size / 2 + 1)))
        y_idx = list(range(int(point.y - window_size / 2), int(point.y + window_size / 2 + 1)))

        return x_idx, y_idx

    def GetWindow(self, point, window_size):
        cols, rows = self.GetIndexes(point, window_size)
        res = self.img[rows]
        return res[:, cols, :]

    @staticmethod
    def CalculateSSD(a, b):
        tmp = np.linalg.norm(np.linalg.norm((a - b), 2, 2), 1)
        return tmp

    @staticmethod
    def ApplyAffineTrans(source_point, M):
        return np.dot(M[:, :2], np.asarray([source_point.x, source_point.y])) + M[:, -1]

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

        for k in range(corners.shape[0]):
            point = self.Point(corners[k, 0], corners[k, 1])
            self.feature_points_vec.append(point)


def initChosenFeatures(corners):
    chosen_features = np.empty((6, 3, 2), dtype=np.float32)

    features_mat = [[1, 6, 13], [1, 9, 18], [1, 8, 14], [3, 10, 18], [1, 6, 13], [6, 13, 19]]

    for k in range(6):
        chosen_features[k, :, :] = corners[k][features_mat[k], :]

    return chosen_features


class SourceFrame(Frame):
    CoupledPoints = namedtuple('CoupledPoints', ['src_point', 'dst_point'])

    def __init__(self, src_img, dst_img_vec):
        super().__init__(0, src_img, src_img)
        self.frame_vec = [Frame(k, pic, src_img) for k, pic in enumerate(dst_img_vec)]
        self.frame_num = len(self.frame_vec)
        self.coupled_points = {k: [] for k in range(self.frame_num)}
        self.affine_mat = {k: np.empty((2, 3)) for k in range(self.frame_num)}
        self.affine_inv_mat = {k: np.empty((2, 3)) for k in range(self.frame_num)}
        self.inv_affine_imgs = {k: np.empty(src_img.shape) for k in range(self.frame_num)}
        self.affine_imgs = {k: np.empty(src_img.shape) for k in range(self.frame_num)}
        self.trajectories = TrajList(self.frame_vec)


    @staticmethod
    def ThrowError():
        raise SourceFrameError

    def calcAffineTrans(self, dst_frame_idx, selected_idx_vec):
        reference_pts = np.zeros((3, 2), dtype=np.float32)
        shifted_pts = np.zeros((3, 2), dtype=np.float32)
        for k, selected_idx in enumerate(selected_idx_vec):
            reference_pts[k, :] = np.round(self.coupled_points[dst_frame_idx][selected_idx].src_point)
            shifted_pts[k, :] = np.round(self.coupled_points[dst_frame_idx][selected_idx].dst_point)

        M = cv.getAffineTransform(reference_pts, shifted_pts)
        iM = np.zeros(M.shape)
        cv.invertAffineTransform(M, iM)
        return M, iM

    @staticmethod
    def SearchFeaturePoints(src_point, L, dst_point_vec):
        points_in_range = []
        for potential_point in dst_point_vec:
            x_dist = abs(potential_point.x - src_point.x)
            y_dist = abs(potential_point.y - src_point.y)
            if np.all(abs(np.asarray([x_dist, y_dist])) <= L / 2):
                points_in_range.append(potential_point)

        return points_in_range

    def FindBestPoint(self, src_point, dst_frame, potential_point_vec, window_size):
        src_window = self.GetWindow(src_point, window_size)

        ssd_vec = []
        for dst_point in potential_point_vec:
            try:
                ssd_vec.append(self.CalculateSSD(src_window, dst_frame.GetWindow(dst_point, window_size)))
            except FrameError:
                dst_frame.feature_points_vec.remove(dst_point)
                continue

        return potential_point_vec[np.argmin([ssd_vec])]

    def AutomaticMatch(self, dst_frame_idx, L, W):
        dst_frame = self.frame_vec[dst_frame_idx]

        for k, src_point in enumerate(self.feature_points_vec):
            try:
                points_in_range = self.SearchFeaturePoints(src_point, L, dst_frame.feature_points_vec)
                if len(points_in_range) == 0:
                    continue

                if len(points_in_range) == 1:
                    best_point = points_in_range[0]
                else:
                    best_point = self.FindBestPoint(src_point, dst_frame, points_in_range, W)

                self.AddCoupledPoints(dst_frame_idx, self.CoupledPoints(src_point, best_point), 255 - 10 * k, k)
            except SourceFrameError:
                self.feature_points_vec.remove(src_point)
                continue

    def AddCoupledPoints(self, dst_frame_idx, coupled_points, color, k):
        self.coupled_points[dst_frame_idx].append(coupled_points)

        self.ZeroPixelsInWindow(coupled_points.src_point, 10, self.frame_vec[dst_frame_idx].reference_img, color)
        self.ZeroPixelsInWindow(coupled_points.dst_point, 10, self.frame_vec[dst_frame_idx].img, color)

    def RANSAC(self, dst_frame_idx):
        biggest_inlier = []
        best_M = np.empty((2, 3))
        best_iM = np.empty((2, 3))
        for _ in range(100):
            rand_pts_idx = permutation(len(self.coupled_points[dst_frame_idx]))[:3]
            M, iM = self.calcAffineTrans(dst_frame_idx, rand_pts_idx)

            src_points_vec = [couple.src_point for couple in self.coupled_points[dst_frame_idx]]
            dst_points_vec = [couple.dst_point for couple in self.coupled_points[dst_frame_idx]]
            source_trans = [self.ApplyAffineTrans(source_point, M) for source_point in src_points_vec]
            points_dist = calcEuclideanDist(source_trans, dst_points_vec)
            inlier_idx = np.where(points_dist < 30)[0]
            inlier_group = [src_points_vec[idx] for idx in inlier_idx]

            if len(inlier_group) > len(biggest_inlier):
                biggest_inlier = inlier_group
                match_points = [source_trans[idx] for idx in inlier_idx]
                best_M = M
                best_iM = iM
                best_dist = points_dist

        self.affine_mat[dst_frame_idx] = best_M
        self.affine_inv_mat[dst_frame_idx] = best_iM

    def applyAffineTrans(self):
        rows, cols, ch = self.img.shape
        for k in range(self.frame_num):
            self.affine_imgs[k] = (cv.warpAffine(self.img, self.affine_mat[i], (cols, rows)))

    def applyInvAffineTrans(self):
        rows, cols, ch = self.img.shape
        for k in range(self.frame_num):
            self.inv_affine_imgs[k] = (cv.warpAffine(self.frame_vec[k].img, self.affine_inv_mat[k], (cols, rows)))

    def CreateTrajectoryMat(self):
        trajectories_list = self.trajectories.trajectory_list
        traj_mat = np.zeros((2 * len(trajectories_list), self.frame_num))
        for trajectory_idx, trajectory in enumerate(trajectories_list):
            for frame_idx in trajectory.keys():
                x_mat_idx = 2 * trajectory_idx
                traj_mat[x_mat_idx, frame_idx] = trajectory[frame_idx].x
                y_mat_idx = x_mat_idx + 1
                traj_mat[y_mat_idx, frame_idx] = trajectory[frame_idx].y

        return traj_mat
''' Aux Methods'''


def plotRconstImg(input_img, output, real):
    plt.subplot(131), plt.imshow(input_img), plt.title('frame')
    plt.subplot(132), plt.imshow(output), plt.title('trans frame')
    plt.subplot(133), plt.imshow(real), plt.title('reference')
    plt.show()


def calcEuclideanDist(estimated_points_list, real_points_list):
    dist_list = [np.linalg.norm(estimated_point - real_point, 2) for estimated_point, real_point in
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

class SourceFrameError(ValueError):
    pass


class FrameError(ValueError):
    pass


def section3(src_frame):
    for frame in src_frame.frame_vec:
        plt.subplot(121), plt.imshow(frame.reference_img), plt.title('reference')
        plt.subplot(122), plt.imshow(frame.img), plt.title('frame')
        plt.show()

def smartStbilization(source_frame, k, delta):
    traj = source_frame.CreateTrajectoryMat()
    broken_mat_list = breakTrajMat(traj, k, delta)



def breakTrajMat(traj_mat, k, delta):
    mat_list = []
    rows, cols = traj_mat.shaepe

    start_frame = 0
    end_frame = delta - 1
    while end_frame < cols:
        new_mat = traj_mat[:, range(start_frame, end_frame + 1)]
        start_frame += delta
        end_frame += delta

        new_mat[(new_mat != 0).all(axis=1)]
        mat_list.append(new_mat)

    return mat_list


if __name__ == '__main__':

    a = np.random.randint(0, 8, (5, 10))
    u, s, vh = np.linalg.svd(a, full_matrices=True)
    tmp = u[:, :6] * s
    np.allclose(a, np.dot(tmp, vh))
    smat = np.zeros((9, 6), dtype=complex)
    smat[:6, :6] = np.diag(s)
    np.allclose(a, np.dot(u, np.dot(smat, vh)))




    # extractImages('pen.mp4', 'extractedImgs')
    # makeVideoMask('extractedImgs')
    # createVideo('extractedImgs', 'masks', 'masked_pen.avi', 30)
    # extractImages('masked_pen.avi', 'masked_extracted')

    frames_num_manual = list(range(20, 100, 15))
    frames_num = list(range(150))
    frames_names = ['extractedImgs/frame' + "%03d" % num + '.jpg' for num in frames_num]
    frames = [cv.imread(im) for im in frames_names]

    source_frame = SourceFrame(frames[0], frames[1:])


    for i in range(source_frame.frame_num):
        source_frame.AutomaticMatch(i, 100, 40)
        source_frame.RANSAC(i)

    # section3(source_frame)

    source_frame.applyAffineTrans()
    source_frame.applyInvAffineTrans()

    # for i in range(source_frame.frame_num):
    #     output = source_frame.affine_imgs[i]
    #     plotRconstImg(source_frame.img, output, source_frame.frame_vec[i].img)
    #
    # for i in range(source_frame.frame_num):
    #     output = source_frame.inv_affine_imgs[i]
    #     plotRconstImg(source_frame.frame_vec[i].img, output, source_frame.img)
    #
    stabilized_img = [source_frame.inv_affine_imgs[i] for i in range(source_frame.frame_num)]
    createVideoFromList(stabilized_img, 'stabi.avi', 20)
    print('all done')
