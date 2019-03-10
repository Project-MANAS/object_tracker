#!/usr/bin/python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from video_generator import VideoGenerator

def main():
    pub = rospy.Publisher('tracking_frames', CompressedImage, queue_size=10)
    rospy.init_node('video_generator')
    rate = rospy.Rate(2)

    generator = VideoGenerator(640, 480, 3, 3, 5, 20, 25)

    while not rospy.is_shutdown():
        frame = generator.generate_frame()
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = 'jpeg'
        encoded = cv2.imencode('.jpg', frame)[1].tostring()
        msg.header.stamp = rospy.Time.now()
        print(msg.header.stamp)
        msg.data = encoded
        pub.publish(msg)
        rate.sleep()

if __name__ == "__main__":
    main()

