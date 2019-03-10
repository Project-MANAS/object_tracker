#!/usr/bin/python3
import rospy
import cv2
import message_filters
import numpy as np
from sensor_msgs.msg import CompressedImage
from objecttracker.msg import DetectedObject, DetectedObjectArray

class ObjectVisualizer:
    def __init__(self, hres=1024, vres=768):
        img_sub = message_filters.Subscriber("tracking_frames", CompressedImage)
        obj_sub = message_filters.Subscriber("tracked_objects", DetectedObjectArray)
        ts = message_filters.TimeSynchronizer([img_sub, obj_sub], 20)
        ts.registerCallback(self.callback)
        self.show_obj_info = False
        rospy.spin()
        
        
    def callback(self, image, objects):
        np_arr = np.fromstring(image.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        detected_objects = objects.objects
        for obj in detected_objects:
            point = (obj.pose.position.x, obj.pose.position.y)
            cv2.circle(img, (int(point[0]), int(point[1])), 5, (255, 0, 0))
            bounding_box = (
                    int(point[0] - obj.dimensions.x/2), 
                    int(point[1] - obj.dimensions.y/2),
                    int(point[0] + obj.dimensions.x/2), 
                    int(point[1] + obj.dimensions.y/2)
            )
            cv2.rectangle(
                img,
                (bounding_box[0], bounding_box[1]),
                (bounding_box[2], bounding_box[3]),
                (0, 255, 0)
            )
            if self.show_obj_info:
                label = "id: " + str(obj.id) \
                        + " vx: " + str(obj.velocity.linear.x) \
                        + " vy: " + str(obj.velocity.linear.y)
                cv2.putText(img, label, (int(point[0]+3), int(point[1]+3)),
                            font, 1, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow("output", img)
        cv2.waitKey(1)

def main():
    ObjectVisualizer(640,480)

if __name__ == "__main__":
    rospy.init_node("visualizer_node")
    main()
