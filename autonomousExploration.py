#!/usr/bin/env python3
# autonomousExploration.py
import rospy
from std_msgs.msg import String, Bool

def autonomousExploration():
    pub = rospy.Publisher('/explore', Bool, queue_size=10)
    rospy.init_node('exploration', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        AutonomousList = input("Hit enter to start autonomous exploration: \n")
        pub.publish(True)
        rate.sleep()

if __name__ == '__main__':
    try:
        autonomousExploration()
    except rospy.ROSInterruptException:
        pass
