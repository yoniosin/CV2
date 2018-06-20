import matplotlib.pyplot as plt
from numpy.random import permutation
from Q3 import *
from klt import *
from sklearn.utils.extmath import randomized_svd as svd_t
# from scipy.signal import savgol_filter
from scipy import ndimage

CoupledPoints = namedtuple('CoupledPoints', ['src_point', 'dst_point'])


class Frame:
    # Frame class, calculate the feature_points_vec by cornerDetector()
    def __init__(self, idx, img):
        self.idx = idx
        self.img = img
        self.feature_points_vec = []
        self.cornerDetector()

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
        return M[:, :2] @ np.asarray([source_point.x, source_point.y]) + M[:, 2]

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

        # save the points in feature_points_vec of Frame
        for k in range(corners.shape[0]):
            point = Point(corners[k, 0], corners[k, 1])
            self.feature_points_vec.append(point)


class SourceFrame(Frame):
    # SourchFrame class: create list of Frames, calculate for each frame the features by harris detector and calculate
    #  list of trajectores
    def __init__(self, src_img, dst_img_vec):
        super().__init__(0, src_img)
        self.frame_vec = [Frame(k, pic) for k, pic in enumerate(dst_img_vec)]
        self.frame_num = len(self.frame_vec)
        self.coupled_points = {k: [] for k in range(self.frame_num)}
        self.affine_mat = {k: np.empty((2, 3)) for k in range(self.frame_num)}
        self.affine_inv_mat = {k: np.empty((2, 3)) for k in range(self.frame_num)}
        self.trajectories = TrajList(self.frame_vec)

    @staticmethod
    def ThrowError():
        raise SourceFrameError

    @staticmethod
    def calcAffineTrans(coupled_points_list, selected_idx_vec):
        reference_pts = np.zeros((3, 2), dtype=np.float32)
        shifted_pts = np.zeros((3, 2), dtype=np.float32)

        # run through the selected idx, build two array of the reference and the shifted points
        for k, selected_idx in enumerate(selected_idx_vec):
            reference_pts[k, :] = coupled_points_list[selected_idx].src_point
            shifted_pts[k, :] = coupled_points_list[selected_idx].dst_point

        # calc the affine and inverse affine matrix between the two points
        M = cv.getAffineTransform(reference_pts, shifted_pts)
        iM = np.zeros(M.shape)
        cv.invertAffineTransform(M, iM)
        return M, iM

    @staticmethod
    def SearchFeaturePoints(src_point, L, dst_point_vec):
        # find all points from dst_point_vec that fit inside  L sized window around the src_point
        points_in_range = []
        for potential_point in dst_point_vec:
            x_dist = abs(potential_point.x - src_point.x)
            y_dist = abs(potential_point.y - src_point.y)
            if np.all(abs(np.asarray([x_dist, y_dist])) <= L / 2):
                points_in_range.append(potential_point)

        return points_in_range

    def FindBestPoint(self, src_point, dst_frame, potential_point_vec, window_size):
        # find the best coupled point for the source point, from list of potential points in the dst frame
        src_window = self.GetWindow(src_point, window_size)

        ssd_vec = []
        # for each of the points in the dst frame, calc ssd in window defined by window_size
        for dst_point in potential_point_vec:
            try:
                ssd_vec.append(self.CalculateSSD(src_window, dst_frame.GetWindow(dst_point, window_size)))
            # if the dst point is to close to the edge, GetWindow raise and we cannot calculate ssd -> remove this point
            except FrameError:
                dst_frame.feature_points_vec.remove(dst_point)
                continue
        try:
            return potential_point_vec[np.argmin([ssd_vec])]
        # if no dst point were able to calculate ssd -> raise SourceFrameError
        except ValueError:
            raise SourceFrameError

    def AutomaticMatch(self, dst_frame_idx, L, W):
        dst_frame = self.frame_vec[dst_frame_idx]

        # for each  point in the source features vec, search for all the points of the dst frame, inside the
        # specified window (L)
        for k, src_point in enumerate(self.feature_points_vec):
            try:
                # search for the relevant points
                points_in_range = self.SearchFeaturePoints(src_point, L, dst_frame.feature_points_vec)

                # if number of points is 0, skip this source point
                if len(points_in_range) == 0:
                    continue

                # if number of points is 1, it is the coupled point for this source point
                if len(points_in_range) == 1:
                    best_point = points_in_range[0]
                else:
                    # there is more than one points in the L-window: choose the best by ssd
                    best_point = self.FindBestPoint(src_point, dst_frame, points_in_range, W)

                self.AddCoupledPoints(dst_frame_idx, src_point, best_point)

            # if noe of the points_in_range were able to calculate ssd -> remove the source point
            except SourceFrameError:
                self.feature_points_vec.remove(src_point)
                continue

    def AddCoupledPoints(self, dst_frame_idx, source_point, dest_point):
        coupled_points = CoupledPoints(source_point, dest_point)
        self.coupled_points[dst_frame_idx].append(coupled_points)

    def runRANSACForFrame(self, dst_frame_idx):
        # extract the coupled point and run RANSAC, save the transformation matrixes
        coupled_points_list = self.coupled_points[dst_frame_idx]
        best_M, best_iM = self.RANSAC(coupled_points_list)

        self.affine_mat[dst_frame_idx] = best_M
        self.affine_inv_mat[dst_frame_idx] = best_iM

    def RANSAC(self, coupled_points_list, iter_num=100, thresh=15):
        biggest_inlier = []
        best_M = np.empty((2, 3))
        best_iM = np.empty((2, 3))
        for _ in range(iter_num):
            # choose 3 coupled points from list and calculate the affine matrix for them
            rand_pts_idx = permutation(len(coupled_points_list))[:3]
            M, iM = self.calcAffineTrans(coupled_points_list, rand_pts_idx)

            # apply the affine transformation on all feature points in the source frame
            src_points_vec = [couple.src_point for couple in coupled_points_list]
            dst_points_vec = [couple.dst_point for couple in coupled_points_list]
            source_trans = [self.applyAffineTransPerPoint(source_point, M) for source_point in src_points_vec]

            # calculate the distance between the transferred source points to the coupled dst point, and choose the
            # points with distance blow thresh
            points_dist = calcEuclideanDist(source_trans, dst_points_vec)
            inlier_idx = np.where(points_dist < thresh)[0]
            inlier_group = [src_points_vec[idx] for idx in inlier_idx]

            # update the biggest inlier group until now
            if len(inlier_group) > len(biggest_inlier):
                biggest_inlier = inlier_group
                best_M = M
                best_iM = iM

        return best_M, best_iM

    def applyAffineTransForAllFrames(self):
        rows, cols, _ = self.img.shape
        affine_imgs = []

        # apply the affine transformation (saved in self.affine_mat) for each of the frames in frame_list
        for k, frame in enumerate(self.frame_vec):
            affine_imgs.append(cv.warpAffine(frame.img, self.affine_mat[k], (cols, rows)))

        return affine_imgs

    def applyInvAffineTransForAllFrames(self, frame_list=None):
        rows, cols, ch = self.img.shape
        frame_list_real = range(self.frame_num) if frame_list is None else frame_list
        inv_affine_imgs = []

        # apply the inverse affine transformation (saved in self.affine_inv_mat) for each of the frames in frame_list
        for k in frame_list_real:
            inv_affine_imgs.append(cv.warpAffine(self.frame_vec[k].img, self.affine_inv_mat[k], (cols, rows)))

        return inv_affine_imgs

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

    @staticmethod
    def calcStabMat(mat, k, r):
        # calc SVD of the mat, tak just r relevant rows/cols
        u, s, vh = np.linalg.svd(mat)
        u = u[:, :r]
        s = s[:r]
        vh = vh[:r, :]

        # create the matrixes that make up 'mat'
        c = u @ np.diag(np.sqrt(s))
        e = np.diag(np.sqrt(s)) @ vh

        # smooth rows of e by gaussian filter - corresponding to smooth the trajectories
        sigma = k / (np.sqrt(2))
        e_stab = ndimage.gaussian_filter(e, (0, sigma))

        # assemble the stabilized new mat
        new_mat = c @ e_stab
        return new_mat

    def smartStabilization(self, k, delta, r):

        # from trajectory list (saved in SourceFrame class), bulid M matrix and break it into kxk matrixes.
        trajectory = self.CreateTrajectoryMat()
        broken_mat_list = breakTrajMat(trajectory, k, delta)
        affine_mat_dict = {}

        # for each mat, calc the stabilized mat (smooth the trajectories) and find the best affine transformation
        # (using RANSAC)
        for mat_num, mat in enumerate(broken_mat_list):
            new_mat = self.calcStabMat(mat, k, r)

            # choose which frames (column of the mat) to calc this iteration
            max_window = delta if mat_num < len(broken_mat_list) - 1 else k
            x_mat, y_mat = ExtractCoorFromM(mat[:, :max_window])
            x_new_mat, y_new_mat = ExtractCoorFromM(new_mat[:, :max_window])

            # run RANSAC for each of the frames
            for column_idx, (x_column, y_column, new_x_column, new_y_column) in \
                    enumerate(zip(x_mat.T, y_mat.T, x_new_mat.T, y_new_mat.T)):
                coupled_points = []
                for j in range(len(x_column)):
                    original_point = Point(x_column[j], y_column[j])
                    smoothed_point = Point(new_x_column[j], new_y_column[j])
                    coupled_points.append(CoupledPoints(original_point, smoothed_point))

                affine_mat, _ = self.RANSAC(coupled_points, 100, 0.5)

                frame_idx = mat_num * delta + column_idx
                affine_mat_dict[frame_idx] = affine_mat

        self.affine_mat = affine_mat_dict


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
    dist_list = [np.linalg.norm(estimated_point - real_point, ord=2) for estimated_point, real_point in
                 zip(estimated_points_list, real_points_list)]
    return np.asarray(dist_list)


def breakTrajMat(traj_mat, k, delta):
    mat_list = []
    rows, cols = traj_mat.shape

    start_frame = 0
    end_frame = k - 1
    while end_frame < cols:
        new_mat = traj_mat[:, range(start_frame, end_frame + 1)]
        start_frame += delta
        end_frame += delta

        new_mat = new_mat[(new_mat != 0).all(axis=1)]
        mat_list.append(new_mat)

    return mat_list


def section2(data_set):
    # for each frame, color the area around each point detected by harris
    for frame in data_set.frame_vec[::20]:
        img_with_harris = np.copy(frame.img)
        for point in frame.feature_points_vec:
            color = np.random.randint(0, 255, (1, 3))
            frame.ZeroPixelsInWindow(point, 5, img_with_harris, color)

        # plot the result
        plt.figure()
        plt.imshow(img_with_harris)
        plt.title('Frame #' + str(frame.idx) + ' Corners Detected using Harris')
        plt.show()


def getPointsForManualMatching():
    ref_point_list = [Point(171, 32), Point(484, 53), Point(194, 170)]
    dst_point_dict = {20: [Point(169, 52), Point(480, 65), Point(194, 184)],
                      40: [Point(205, 78), Point(513, 79), Point(233, 205)],
                      60: [Point(188, 56), Point(488, 68), Point(211, 182)],
                      80: [Point(160, 34), Point(462, 55), Point(184, 165)],
                      100: [Point(219, 25), Point(517, 38), Point(241, 153)],
                      120: [Point(190, 74), Point(491, 84), Point(217, 202)],
                      }
    return ref_point_list, dst_point_dict


def Manual(data_set):
    # define the 3 coupled point from source - the reference image to dst - each of the frames
    ref_point_list, dst_point_dict = getPointsForManualMatching()

    # for each of the frames, create list of CoupledPoints from the pair of points above
    for frame_idx in dst_point_dict.keys():
        dst_point_list = dst_point_dict[frame_idx]
        coupled_points = [CoupledPoints(ref_point, dst_point) for ref_point, dst_point
                          in zip(ref_point_list, dst_point_list)]

        # calculate the affine transformation between he coupled points
        M, iM = data_set.calcAffineTrans(coupled_points, [0, 1, 2])
        data_set.affine_inv_mat[frame_idx] = iM

    # apply the inverse transformation on the relevant frames
    output = data_set.applyInvAffineTransForAllFrames(dst_point_dict.keys())

    # plot the result
    for i, frame_idx in enumerate(dst_point_dict.keys()):
        plotRconstImg(data_set.frame_vec[frame_idx].img, output[i], data_set.img, frame_idx)

    # create video from the result (video is short, so we take 2 fps)
    createVideoFromList(output, 'manual_stabilized.avi', 2)


def section6(data_set):
    # perform automatic match for all frames
    for frame in data_set.frame_vec:
        source_frame.AutomaticMatch(frame.idx, 100, 50)

    # plot the pair images of the reference image and each of the frames with relevant matched points
        dst_img_disp = np.copy(frame.img)
        ref_img_disp = np.copy(data_set.img)
        for coupled_points in data_set.coupled_points[frame.idx][:10]:
            color = np.random.randint(0, 255, (1, 3))
            data_set.ZeroPixelsInWindow(coupled_points.src_point, 5, ref_img_disp, color)
            data_set.ZeroPixelsInWindow(coupled_points.dst_point, 5, dst_img_disp, color)

        if frame.idx in list(range(0, 121, 20)):
            plt.figure()
            plt.subplot(121), plt.imshow(ref_img_disp), plt.title('Automatic matched points - reference image')
            plt.subplot(122), plt.imshow(dst_img_disp), plt.title('Frame #' + str(frame.idx))
            plt.show()


def section7(data_set):
    # perform automatic match for all frames
    for i in range(source_frame.frame_num):
        data_set.runRANSACForFrame(i)


def section8(data_set):
    # apply the inverse affine transformation for all frames, plot the result and create video o the stabilized video
    inv_affine_imgs = data_set.applyInvAffineTransForAllFrames()

    for i in range(0, 121, 20):
        plotRconstImg(source_frame.frame_vec[i].img, inv_affine_imgs[i], source_frame.img, i)

    stabilized_img = [inv_affine_imgs[i] for i in range(data_set.frame_num)]
    createVideoFromList(stabilized_img, 'stabilized_vid.avi', 30)


def section9(data_set):
    # calc thesmart stabilizatio by the article, apply the affine transformation for all frames and create video of
    # the stabilized video
    data_set.smartStabilization(50, 1, 9)
    tmp = data_set.applyAffineTransForAllFrames()
    createVideoFromList(tmp, 'smart_stabilized_vid.avi', 30)


if __name__ == '__main__':
    # create the directory to store frames if it doesn't exists
    directory_name = 'extractedImgs'
    # if not os.path.exists(directory_name):
    #     os.makedirs(directory_name)
    # extract the images from video to frames
    # extractImages('inv.mp4', directory_name)

    # read the relevant frames
    frames_num = list(range(151))
    frames_names = [directory_name + '/frame' + "%03d" % num + '.jpg' for num in frames_num]
    frames = [cv.imread(im)[:, :, ::-1] for im in frames_names]

    # create the data base: SourceFrame class, which have the reference image and list of the Frames class
    # constructor of Frame calculate the points of harris corner detector
    source_frame = SourceFrame(frames[0], frames[1:])

    Manual(source_frame)

    section2(source_frame)
    section6(source_frame)
    section7(source_frame)
    section8(source_frame)
    section9(source_frame)

    print('all done')
