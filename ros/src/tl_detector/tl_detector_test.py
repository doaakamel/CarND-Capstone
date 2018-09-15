#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from scipy.spatial import KDTree

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector_test', log_level=rospy.DEBUG) #log_level=rospy.DEBUG

        self.camera_image = None
        self.lights = []


        sub = rospy.Subscriber('/image_raw', Image, self.image_cb, queue_size=1)
        self.tl_detected_image_pub = rospy.Publisher("detected_tl_image", Image, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()

        rospy.spin()


    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        self.get_light_state()


    def get_light_state(self):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        #return self.light_classifier.get_classification(cv_image)
        tl_cv_image = self.light_classifier.get_classification(cv_image)
        try:
            self.tl_detected_image_pub.publish(self.bridge.cv2_to_imgmsg(tl_cv_image, "bgr8"))
        except CvBridgeError as e:
           print(e)


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
