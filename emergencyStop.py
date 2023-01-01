#!/usr/bin/env python3
# emergencyStop.py
import rospy
from std_msgs.msg import String, Bool

def emergencyStop():
    pub = rospy.Publisher('/emergency_stop', Bool, queue_size=10)
    rospy.init_node('emergency', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        rescueList = input("Hit enter to stop the robot in case of emergency: \n")
        pub.publish(True)
        rate.sleep()

if __name__ == '__main__':
    try:
        emergencyStop()
    except rospy.ROSInterruptException:
        pass


