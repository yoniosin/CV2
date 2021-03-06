import cv2
import numpy as np
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])


class TrajList:
    # Trajector list of all the frames, using klt
    def __init__(self, frames_vec):
        self.trajectory_list = []
        self.klt(frames_vec)

    def addTrajPoint(self, frame_idx, p0, p1):
        p0 = Point(p0[0, 1], p0[0, 0])
        p1 = Point(p1[0, 1], p1[0, 0])

        # search for trajectory that end with the last point
        for trajectory in self.trajectory_list:
            if (frame_idx - 1) in trajectory.keys() and trajectory[frame_idx - 1] == p0:
                trajectory[frame_idx] = p1
                return

        # if there is no trajectory like that, create new trajectory with both points in it
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

        # Take first frame and find corners in it
        old_frame = frame_vec[0].img
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

        # for each of the frames, find corners and try to adjust them to the pints of the last frame
        for frame_idx in range(1, len(frame_vec)):
            frame = frame_vec[frame_idx].img
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # add the point to the trajectory list
            for st_idx in np.where(st == 1)[0]:
                self.addTrajPoint(frame_idx, p0[st_idx], p1[st_idx])

            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = p1
