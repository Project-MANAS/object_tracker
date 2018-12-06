import cv2
import numpy as np


class ObjectTracker:
    def __init__(self):
        self.clustering = None
        self.objects = None

    def get_contour_centers(img):
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [np.mean(contour, axis=0) for contour in contours]

    def get_next_state_predictions(self):
        raise NotImplementedError

    def assign_object_ids(self, centers):
        raise NotImplementedError

    def compute_velocities(self, tagged_centers):
        raise NotImplementedError

    def get_velocities(self):
        raise NotImplementedError

    def get_centers(self):
        raise NotImplementedError

    def add_frame_to_tracker(self, img):
        centers = ObjectTracker.get_contour_centers(img)
        tagged_centers = self.assign_object_ids(centers)
        self.compute_velocities(tagged_centers)


def main():
    orig_img = cv2.imread('/home/chaitanya/Downloads/circles.jpg')
    img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    thresh, img = cv2.threshold(img, 127, 255, 0)
    tracker = ObjectTracker()
    centers = list(ObjectTracker.get_contour_centers(img))
    centers = np.array(centers).squeeze()

    for point in centers:
        cv2.circle(orig_img, (int(point[0]), int(point[1])), 5, (255,0,0))

    cv2.imshow("output", orig_img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
