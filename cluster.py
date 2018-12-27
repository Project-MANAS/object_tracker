import sys
import cv2
import numpy as np


class ObjectTracker:
    def __init__(self):
        self.clustering = None
        self.objects = None
        self.trackers = []

    @staticmethod
    def get_contour_centers(img):
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [np.mean(contour, axis=0) for contour in contours]

    @staticmethod
    def dist(p1, p2):
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

    @staticmethod
    def assign_object_ids(centers, object_next_state):
        annotated_centers = []

        for object in object_next_state:
            object_center_idx = 0
            for i, center in enumerate(centers):
                if ObjectTracker.dist(object[1], center) < ObjectTracker.dist(object[1], centers[object_center_idx]):
                    object_center_idx = i
            annotated_centers.append((object[0],centers[object_center_idx]))

        return annotated_centers

    def add_object(self, center):
        kalman_filter = cv2.KalmanFilter(4,2,0)
        kalman_filter.transitionMatrix = np.array([[1,0,1,0],
                                            [0,1,0,1],
                                            [0,0,1,0],
                                            [0,0,0,1]])
        kalman_filter.measurementMatrix = np.array([[1., 0., 0., 0.],
                                             [0., 1., 0., 0.]])
        kalman_filter.processNoiseCov = 1e-5 * np.eye(4)
        kalman_filter.measurementNoiseCov = 1e-1 * np.ones((1,2))
        kalman_filter.errorCovPost = 1. * np.ones((4,4))
        kalman_filter.statePost = 0.1 * np.randn(4,1)
        kalman_filter.correct(list(center))
        self.trackers.append(kalman_filter)


    def get_next_state_predictions(self):
        raise NotImplementedError

    def update_tracked_objects(self, annotated_centers, object_next_states):
        self.objects = [tuple(list(center) + [0]) for center in annotated_centers]

    def compute_velocities(self, tagged_centers):
        raise NotImplementedError

    def get_objects(self):
        return self.objects

    def track_frame(self, img):
        centers = ObjectTracker.get_contour_centers(img)
        tagged_centers = []
        next_state_predictions = None
        if self.objects is not None:
            next_state_predictions = self.get_next_state_predictions()
            tagged_centers = ObjectTracker.assign_object_ids(centers, next_state_predictions)
        else:
            tagged_centers = list(enumerate(centers))

        self.update_tracked_objects(tagged_centers, next_state_predictions)


def main():
    try:
        vid = cv2.VideoCapture(sys.argv[1])
        print(sys.argv[1])
    except IndexError:
        print("path to video not provided")
        return
    ret = True
    while vid.isOpened() and ret is True:
        ret, orig_img = vid.read()
        img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        thresh, img = cv2.threshold(img, 127, 255, 0)
        tracker = ObjectTracker()
        tracker.track_frame(img)
        centers = [tuple(obj[1][0]) for obj in tracker.get_objects()]

        for point in centers:
            if point is None:
                continue
            cv2.circle(orig_img, (int(point[0]), int(point[1])), 5, (255,0,0))

        cv2.imshow("output", orig_img)
        cv2.waitKey(100)
    else:
        print("Unable to open video")



if __name__ == "__main__":
    main()