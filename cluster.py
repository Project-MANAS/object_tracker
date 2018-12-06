import cv2
import numpy as np
import sklearn
import sklearn.cluster


def silhouette_coefficient():
    raise NotImplementedError


class ObjectTracker:
    def __init__(self):
        self.clustering = None
        self.centers = None

    def find_clusters(self, img):
        clustering = sklearn.cluster.DBSCAN(eps=0.5, min_samples=5).fit(img)
        if self.clustering is None:
            self.clustering = clustering
        
        num_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        cluster_centers_x = np.zeros(num_clusters)
        cluster_centers_y = np.zeros(num_clusters)

        for idx, label in enumerate(clustering.labels_):
            if label == -1:
                continue
            cluster_centers_x[label] += idx
            cluster_centers_y[label] += idx

        self.clustering = clustering
        return zip(cluster_centers_x, cluster_centers_y)

    def get_countour_centers(self, img):
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = [np.mean(contour, axis=0) for contour in contours]
        #print(np.array(contours).shape)
        #print(contours)
        #print(len(hierarchy))
        if self.centers is not None:
            self.centers = centers
        return centers


orig_img = cv2.imread('/home/chaitanya/Downloads/circles.jpg')
img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
thresh, img = cv2.threshold(img, 127, 255, 0)
tracker = ObjectTracker()
centers = list(tracker.get_countour_centers(img))
centers = np.array(centers).squeeze()
#print(centers)

for point in centers:
    cv2.circle(orig_img, (int(point[0]), int(point[1])), 5, (255,0,0))

cv2.imshow("output", orig_img)
cv2.waitKey(0)