import matplotlib.pyplot as plt
from numpy.random import permutation
from Q3 import *
from klt import *
from sklearn.utils.extmath import randomized_svd as svd_t
from scipy.signal import savgol_filter

CoupledPoints = namedtuple('CoupledPoints', ['src_point', 'dst_point'])


class Frame:

    def __init__(self, idx, img, reference_img):
        self.idx = idx
        self.img = img
        self.feature_points_vec = []
        self.reference_img = reference_img
        self.cornerDetector()

        self.img_with_harris = np.copy(self.img)
        for point in self.feature_points_vec:
            color = np.random.randint(0, 255, (1, 3))
            self.ZeroPixelsInWindow(point, 5, self.img_with_harris, color)

        self.manual_matching_mat = np.zeros((3, 2))

    def ZeroPixelsInWindow(self, center, window_size, img, color):
        try:
            x_idx_vec, y_idx_vec = self.GetIndexes(center, window_size)
        except ValueError:
            return
        for x_idx in x_idx_vec:
            for y_idx in y_idx_vec:
                img[y_idx, x_idx] = np.uint8(color)

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
        a = a.astype(int)
        b = b.astype(int)
        sub = a - b
        pixel_norm = np.linalg.norm(sub, axis=2)
        return np.sum(pixel_norm)

    @staticmethod
    def applyAffineTransPerPoint(source_point, M):
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
            point = Point(corners[k, 0], corners[k, 1])
            self.feature_points_vec.append(point)


class SourceFrame(Frame):

    def __init__(self, src_img, dst_img_vec):
        super().__init__(0, src_img, src_img)
        self.frame_vec = [Frame(k, pic, np.copy(src_img)) for k, pic in enumerate(dst_img_vec)]
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

    @staticmethod
    def calcAffineTrans(coupled_points_list, selected_idx_vec):
        reference_pts = np.zeros((3, 2), dtype=np.float32)
        shifted_pts = np.zeros((3, 2), dtype=np.float32)
        for k, selected_idx in enumerate(selected_idx_vec):
            reference_pts[k, :] = np.round(coupled_points_list[selected_idx].src_point)
            shifted_pts[k, :] = np.round(coupled_points_list[selected_idx].dst_point)

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
        try:
            return potential_point_vec[np.argmin([ssd_vec])]
        except ValueError:
            raise SourceFrameError

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

                self.AddCoupledPoints(dst_frame_idx, src_point, best_point, k)
            except SourceFrameError:
                self.feature_points_vec.remove(src_point)
                continue

    def AddCoupledPoints(self, dst_frame_idx, source_point, dest_point, pnt_num):
        coupled_points = CoupledPoints(source_point, dest_point)
        self.coupled_points[dst_frame_idx].append(coupled_points)

        if pnt_num < 10:
            color = np.random.randint(0, 255, (1, 3))
            self.ZeroPixelsInWindow(coupled_points.src_point, 5, self.frame_vec[dst_frame_idx].reference_img, color)
            self.ZeroPixelsInWindow(coupled_points.dst_point, 5, self.frame_vec[dst_frame_idx].img, color)

    def runRANSACForFrame(self, dst_frame_idx):
        coupled_points_list = self.coupled_points[dst_frame_idx]
        best_M, best_iM = self.RANSAC(coupled_points_list)

        self.affine_mat[dst_frame_idx] = best_M
        self.affine_inv_mat[dst_frame_idx] = best_iM

    def RANSAC(self, coupled_points_list, iter_num=100, thresh=15):
        biggest_inlier = []
        best_M = np.empty((2, 3))
        best_iM = np.empty((2, 3))
        for _ in range(iter_num):
            rand_pts_idx = permutation(len(coupled_points_list))[:3]
            M, iM = self.calcAffineTrans(coupled_points_list, rand_pts_idx)

            src_points_vec = [couple.src_point for couple in coupled_points_list]
            dst_points_vec = [couple.dst_point for couple in coupled_points_list]
            source_trans = [self.applyAffineTransPerPoint(source_point, M) for source_point in src_points_vec]
            points_dist = calcEuclideanDist(source_trans, dst_points_vec)
            inlier_idx = np.where(points_dist < thresh)[0]
            inlier_group = [src_points_vec[idx] for idx in inlier_idx]

            if len(inlier_group) > len(biggest_inlier):
                biggest_inlier = inlier_group
                best_M = M
                best_iM = iM

        return best_M, best_iM

    def applyAffineTransForAllFrames(self):
        rows, cols, ch = self.img.shape
        for k in range(self.frame_num):
            self.affine_imgs[k] = (cv.warpAffine(self.img, self.affine_mat[k], (cols, rows)))

    def applyInvAffineTransForAllFrames(self, frame_list=None):
        rows, cols, ch = self.img.shape
        frame_list_real = range(self.frame_num) if frame_list is None else frame_list
        for k in frame_list_real:
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

    def smartStabilization(self, k, delta, r):
        trajectory = self.CreateTrajectoryMat()
        broken_mat_list = breakTrajMat(trajectory, k, delta)
        inv_affine_mat_dict = {}

        for mat_num, mat in enumerate(broken_mat_list):
            u, s, vh = svd_t(mat, n_components=r)
            c = np.matmul(u, np.diag(s))
            e = np.zeros((r, k))

            for m in range(r):
                e[m, :] = savgol_filter(vh[m, :], 21, 1)

            max_window = delta if mat_num < len(broken_mat_list) - 1 else k
            new_mat = np.dot(c, e)[:, :max_window]
            x_mat, y_mat = ExtractCoorFromM(mat[:, :max_window])
            x_new_mat, y_new_mat = ExtractCoorFromM(new_mat)

            for column_idx, (x_column, y_column, new_x_column, new_y_column) in \
                    enumerate(zip(x_mat.T, y_mat.T, x_new_mat.T, y_new_mat.T)):
                coupled_points = []
                for j in range(len(x_column)):
                    original_point = Point(x_column[j], y_column[j])
                    smoothed_point = Point(new_x_column[j], new_y_column[j])
                    coupled_points.append(CoupledPoints(original_point, smoothed_point))

                _, inv_affine_mat = self.RANSAC(coupled_points)

                frame_idx = mat_num * delta + column_idx
                inv_affine_mat_dict[frame_idx] = inv_affine_mat

        self.affine_inv_mat = inv_affine_mat_dict


def ExtractCoorFromM(mat):
    return mat[::2, :], mat[1::2, :]


class SourceFrameError(ValueError):
    pass


class FrameError(ValueError):
    pass


''' Aux Methods'''


def initChosenFeatures(corners):
    chosen_features = np.empty((6, 3, 2), dtype=np.float32)

    features_mat = [[1, 6, 13], [1, 9, 18], [1, 8, 14], [3, 10, 18], [1, 6, 13], [6, 13, 19]]

    for k in range(6):
        chosen_features[k, :, :] = corners[k][features_mat[k], :]

    return chosen_features


def plotRconstImg(input_img, output, real, idx):
    plt.figure()
    plt.subplot(131), plt.imshow(input_img), plt.title('frame #' + str(idx))
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


def breakTrajMat(traj_mat, k, delta):
    mat_list = []
    rows, cols = traj_mat.shape

    start_frame = 0
    end_frame = k - 1
    while end_frame <= cols:
        new_mat = traj_mat[:, range(start_frame, end_frame + 1)]
        start_frame += delta
        end_frame += delta

        new_mat[(new_mat != 0).all(axis=1)]
        mat_list.append(new_mat)

    return mat_list


def section2(data_set):
    for frame in data_set.frame_vec[::20]:
        plt.figure()
        plt.imshow(frame.img_with_harris)
        plt.title('Frame #' + str(frame.idx) + ' Corners Detected using Harris')
        plt.show()


def Manual(data_set):
    ref_point_list = [Point(171, 32), Point(484, 53), Point(194, 170)]
    dst_point_dict = {20: [Point(169, 52), Point(480, 65), Point(194, 184)],
                      40: [Point(205, 78), Point(513, 79), Point(233, 205)],
                      60: [Point(188, 56), Point(488, 68), Point(211, 182)],
                      80: [Point(160, 34), Point(462, 55), Point(184, 165)],
                      100: [Point(219, 25), Point(517, 38), Point(241, 153)],
                      120: [Point(190, 74), Point(491, 84), Point(217, 202)],
                      }

    for frame_idx in dst_point_dict.keys():
        dst_point_list = dst_point_dict[frame_idx]
        coupled_points = [CoupledPoints(ref_point, dst_point) for ref_point, dst_point
                          in zip(ref_point_list, dst_point_list)]

        M, iM = data_set.calcAffineTrans(coupled_points, [0, 1, 2])

        data_set.affine_inv_mat[frame_idx] = iM

        data_set.applyInvAffineTransForAllFrames([0, 20, 40, 60, 80, 100, 120])

    output = []
    for i in range(20, data_set.frame_num, 20):
        output.append(data_set.inv_affine_imgs[i])
        plotRconstImg(data_set.frame_vec[i].img, data_set.inv_affine_imgs[i], data_set.img, i)

    createVideoFromList(output, 'manual_stabilized.avi', 2)


def section6(data_set):
    # perform automatic match for all frames
    for i in range(source_frame.frame_num):
        source_frame.AutomaticMatch(i, 100, 50)

    # plot the pair images of the reference image and each of the frames with relevant matched points
    for frame in data_set.frame_vec[::20]:
        plt.figure()
        plt.subplot(121), plt.imshow(frame.reference_img), plt.title('Automatic matched points - reference image')
        plt.subplot(122), plt.imshow(frame.img), plt.title('Frame #' + str(frame.idx))
        plt.show()


def section7(data_set):
    # perform automatic match for all frames
    for i in range(source_frame.frame_num):
        data_set.runRANSACForFrame(i)


def section8(data_set):
    data_set.applyInvAffineTransForAllFrames()

    for i in range(0, source_frame.frame_num, 20):
        output = source_frame.inv_affine_imgs[i]
        plotRconstImg(source_frame.frame_vec[i].img, output, source_frame.img, i)

    stabilized_img = [source_frame.inv_affine_imgs[i] for i in range(data_set.frame_num)]
    createVideoFromList(stabilized_img, 'stabilized_vid.avi', 20)


def section9(data_set):
    data_set.smartStabilization(50, 5, 9)
    data_set.applyInvAffineTransForAllFrames()

    for i in range(0, source_frame.frame_num, 20):
        output = source_frame.inv_affine_imgs[i]
        plotRconstImg(source_frame.frame_vec[i].img, output, source_frame.img, i)

    smart_stabilized_img = [source_frame.inv_affine_imgs[i] for i in range(data_set.frame_num)]
    createVideoFromList(smart_stabilized_img, 'smart_stabilized_vid.avi', 20)


if __name__ == '__main__':
    # extract the images fro video to frames
    extractImages('inv.mp4', 'extractedImgs')

    # create data-set
    frames_num = list(range(151))
    frames_names = ['extractedImgs/frame' + "%03d" % num + '.jpg' for num in frames_num]
    frames = [cv.imread(im)[:, :, ::-1] for im in frames_names]
    source_frame = SourceFrame(frames[0], frames[1:])

    # Manual(source_frame)

    # section2(source_frame)
    # section6(source_frame)
    # section7(source_frame)
    # section8(source_frame)
    section9(source_frame)

    print('all done')
