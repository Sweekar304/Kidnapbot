#!/usr/bin/env python3
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Twist, Pose2D, PoseStamped

x_g = None
y_g = None
theta_g = None

def goal_pose_marker_callback(data):
    """
    loads in goal if different from current goal, and replans
    """
    global x_g
    global y_g
    global theta_g
    if (
        data.x != x_g
        or data.y != y_g
        or data.theta != theta_g
    ):
        print("entered Marker Callback")
        rospy.logdebug(f"New command nav received:\n{data}")
        x_g = data.x
        y_g = data.y
        theta_g = data.theta
        #replan()





def publisher():


    rospy.Subscriber("/goal_pose_marker", Pose2D, goal_pose_marker_callback)

    vis_pub = rospy.Publisher('/marker_topic', Marker, queue_size=10)
    rospy.init_node('marker_node', anonymous=True)
    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        marker = Marker()

        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time()

        # IMPORTANT: If you're creating multiple markers, 
        #            each need to have a separate marker ID.
        marker.id = 0

        marker.type = 2 # sphere
        marker.pose.position.x = x_g
        marker.pose.position.y = y_g
        marker.pose.position.z = 0
        if not x_g == None and not y_g == None:
            marker.pose.position.x = x_g
            marker.pose.position.y = y_g
            marker.pose.position.z = 0
            print("Entered to xg yg")
        else:
            print("Not entered")

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2

        marker.color.a = 1.0 # Don't forget to set the alpha!
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        
        if x_g!=None:
            vis_pub.publish(marker)
            print('Published marker!')
        
        rate.sleep()


if __name__ == '__main__':
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass