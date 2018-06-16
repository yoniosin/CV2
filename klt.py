import cv2
import numpy as np
from collections import namedtuple


class TrajList:
    Point = namedtuple('Point', ['x', 'y'])

    def __init__(self, frames_vec):
        self.trajectory_list = []
        self.klt(frames_vec)

    def addTrajPoint(self, frame_idx, p0, p1):
        p0 = self.Point(p0[0, 0], p0[0, 1])
        p1 = self.Point(p1[0, 0], p1[0, 1])

        for trajectory in self.trajectory_list:
            if (frame_idx - 1) in trajectory.keys() and trajectory[frame_idx - 1] == p0:
                trajectory[frame_idx] = p1
                return

        raise ValueError

    def addNewTraj(self, frame_idx, p0, p1):
        self.trajectory_list.append({frame_idx - 1: p0, frame_idx: p1})

    @staticmethod
    def isPointsEqual(p0, p1):
        if p0.x == p1.x and p0.y == p1.y:
            return True
        else:
            return False

    def klt(self, frame_vec):
        # params for ShiTomasi corner detection
        feature_params = dict(maxCorners=100,
                              qualityLevel=0.3,
                              minDistance=7,
                              blockSize=7)
        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # Create some random colors
        color = np.random.randint(0, 255, (100, 3))
        # Take first frame and find corners in it
        old_frame = frame_vec[0].img
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        last_new_point = np.ones(len(p0))
        points_num = 0

        for frame_idx in range(1, len(frame_vec)):
            frame = frame_vec[frame_idx].img
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            for st_idx in np.where(st == 1)[0]:
                try:
                    self.addTrajPoint(frame_idx, p0[st_idx], p1[st_idx])
                except ValueError:
                    self.addNewTraj(frame_idx, p0[st_idx], p1[st_idx])

            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = p1
