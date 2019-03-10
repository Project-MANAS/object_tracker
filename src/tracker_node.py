#!/usr/bin/python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from object_tracker import ObjectTracker
from objecttracker.msg import DetectedObject, DetectedObjectArray

class TrackerNode:
    def __init__(self, hres=1024, vres=768):
        self.tracker = ObjectTracker(x_max=hres, y_max=vres, distance_threshold=50)
        rospy.Subscriber("tracking_frames", CompressedImage, self.callback)
        self.publisher = rospy.Publisher("tracked_objects", DetectedObjectArray, queue_size=10)
        rospy.spin()
    
    def callback(self, data):
        np_arr = np.fromstring(data.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        thresh, img = cv2.threshold(img, 127, 255, 0)
        self.tracker.track_frame(img)
        detected_objects = []
        objects = self.tracker.get_objects()
        for obj in objects:
            detected_object = DetectedObject()
            detected_object.id = obj.id
            detected_object.pose.position.x = obj.state[0]
            detected_object.pose.position.y = obj.state[1]
            detected_object.pose.position.z = 0
            detected_object.pose.orientation.w = 1
            detected_object.velocity.linear.x = obj.state[2]
            detected_object.velocity.linear.y = obj.state[3]
            detected_object.velocity.linear.z = 0
            detected_object.velocity.angular.x = 0
            detected_object.velocity.angular.y = 0
            detected_object.velocity.angular.z = 0
            detected_object.variance.x = obj.tracker.errorCovPost[0, 0]
            detected_object.variance.y = obj.tracker.errorCovPost[1, 1]
            detected_object.variance.z = 0
            if obj.bounding_box:
                detected_object.dimensions.x = obj.bounding_box[2]
                detected_object.dimensions.y = obj.bounding_box[3]
            detected_object.dimensions.z = 0
            detected_objects.append(detected_object)
        # Add to array of detected objects
        # Convert to message
        detected_obj_array = DetectedObjectArray()
        detected_obj_array.objects = detected_objects
        detected_obj_array.header.stamp = data.header.stamp
        self.publisher.publish(detected_obj_array)


def main():
    rospy.init_node('tracker_node')
    TrackerNode(640,480)


if __name__ == "__main__":
    main()
