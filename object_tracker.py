import sys
import getopt
import time
import cv2
import numpy as np
import scipy.optimize
import collections

TrackedObject = collections.namedtuple("TrackedObject", ["id", "state", "tracker"])


class ObjectTracker:
    def __init__(self, x_min=0, y_min=0, x_max=0, y_max=0, distance_threshold=50, verbose=False):
        self.verbose = verbose
        self.objects = []
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.threshold = distance_threshold

    @staticmethod
    def get_contour_centers(img):

        """
        Given a binary image with white filled in object silhouettes on a black background,
        the function finds and returns centers of these objects as a list

        :param img: A single channel image
        :return: A list of object centers
        """

        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [np.squeeze(np.mean(contour, axis=0)) for contour in contours]

    @staticmethod
    def dist(p1, p2):
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

    @staticmethod
    def create_object(center):

        """
        Add a previously unknown object to the tracker
        :param center: The center of the new object
        :return: None
        """

        kalman_filter = cv2.KalmanFilter(4, 2, 0, cv2.CV_64F)
        kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                                   [0, 1, 0, 1],
                                                   [0, 0, 1, 0],
                                                   [0, 0, 0, 1]], np.float64)
        kalman_filter.measurementMatrix = np.array([[1., 0., 0., 0.],
                                                    [0., 1., 0., 0.]
                                                    ], np.float64)
        kalman_filter.processNoiseCov = 1e-5 * np.eye(4, dtype=np.float64)
        kalman_filter.measurementNoiseCov = 1e-3 * np.eye(2, dtype=np.float64)
        kalman_filter.errorCovPost = 1. * np.ones((4, 4), np.float64)
        kalman_filter.statePost = 0.1 * np.random.randn(4, 1)
        init_state = np.array(list(center) + [0.0, 0.0], np.float64)  # [x, y] + [dx, dy]
        kalman_filter.statePre = init_state.transpose()
        kalman_filter.correct(np.array(center, dtype=np.float64))
        obj_id = time.time()
        return TrackedObject(obj_id, init_state, kalman_filter)

    def get_next_state_predictions(self):
        predicted_objects = []
        for obj in self.objects:
            prediction = obj.tracker.predict()
            predicted_objects.append(TrackedObject(obj.id, np.transpose(prediction), obj.tracker))
        # print(predicted_objects)
        return predicted_objects

    def is_within_tracking_frame(self, point):
        if point[0] < self.x_min \
                or point[0] > self.x_max \
                or point[1] < self.y_min \
                or point[1] > self.y_max:
            return False
        return True

    def update_tracked_objects(self, centers, predicted_objects):
        """
        Given a list of detected centers and predicted states of tracked objects,
        this function assigns detected centers to known trackers when they're within
        a threshold distance and then:
            1) Updates states of currently known objects still within the tracking frame
            2) Deletes trackers for objects that have moved out of the tracking frame
            3) Creates trackers for objects that have wandered into the tracking frame

        :param centers: List of centers detected in the image
        :param predicted_objects: List of predicted states of currently tracked objects
        :return: None
        """
        updated_objects = []
        centers = [center for center in centers if self.is_within_tracking_frame(center)]

        dist_matrix = np.zeros((len(predicted_objects), len(centers)))

        for i, obj in enumerate(predicted_objects):
            for j, center in enumerate(centers):
                dist_matrix[i][j] = ObjectTracker.dist(obj.state[0:2], center)
        matches = scipy.optimize.linear_sum_assignment(dist_matrix)
        matched_objects = matches[0]
        matched_centers = matches[1]
        matches = list(zip(*matches))

        for m in matches:
            if dist_matrix[m[0], m[1]] < self.threshold:
                estimate = predicted_objects[m[0]].tracker.correct(centers[m[1]])
                if self.verbose:
                    print("Center: " + str(centers[m[1]]))
                    print("Estimate: " + str(estimate))

                updated_objects.append(TrackedObject(
                    predicted_objects[m[0]].id,
                    estimate,
                    predicted_objects[m[0]].tracker
                ))
            else:
                print("Deleting object beyond threshold: " + str(predicted_objects[m[0]]))

        for i, obj in enumerate(predicted_objects):
            if matched_objects is None or i not in matched_objects:
                print("Deleting unmatched object: " + str(obj.id))

        for i, center in enumerate(centers):
            if matched_centers is None or i not in matched_centers:
                print("Adding new object for center: " + str(center))
                updated_objects.append(ObjectTracker.create_object(center))

        self.objects = updated_objects

    def get_objects(self):
        return self.objects

    def track_frame(self, img):

        """
        Track the position and velocity of objects in the given frame
        :param img: A binary, single channel image
        :return: None
        """

        centers = ObjectTracker.get_contour_centers(img)
        next_state_predictions = self.get_next_state_predictions()
        self.update_tracked_objects(centers, next_state_predictions)


def print_usage():
    print("Usage: object_tracker.py -i <input file> [OPTIONS]")


def main(argv):
    verbose = False
    infile = ""
    show_obj_info = False

    try:
        opts, args = getopt.getopt(argv, "hvsi:", ["help", "verbose", "show-info", "input="])
    except getopt.GetoptError:
        print_usage()
        return

    for opt, arg in opts:
        if opt=='-i' or opt=='--input':
            infile = arg
        elif opt=='-v' or opt=='--verbose':
            verbose = True
        elif opt=='-h' or opt=='--help':
            print_usage()
        elif opt=='-s' or opt=='--show-info':
            show_obj_info = True

    if infile:
        vid = cv2.VideoCapture(infile)
        print(infile)
    else:
        print("path to video not provided")
        return

    tracker = ObjectTracker(x_max=1013, y_max=757, verbose=verbose)
    font = cv2.FONT_HERSHEY_PLAIN

    while vid.isOpened():
        if verbose:
            print("Reading...")
        ret, orig_img = vid.read()
        if ret is False:
            break
        img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        thresh, img = cv2.threshold(img, 127, 255, 0)
        tracker.track_frame(img)

        for obj in tracker.get_objects():
            point = tuple(obj.state[0:2])
            cv2.circle(orig_img, (int(point[0]), int(point[1])), 5, (255, 0, 0))
            if show_obj_info:
                label = "id: " + str(obj.id) \
                        + " vx: " + str(obj.state[2]) \
                        + " vy: " + str(obj.state[3])
                cv2.putText(orig_img, label, (int(point[0]+3), int(point[1]+3)),
                            font, 1, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("output", orig_img)
        cv2.waitKey(50)
    else:
        print("Unable to open video")


if __name__ == "__main__":
    main(sys.argv[1:])
