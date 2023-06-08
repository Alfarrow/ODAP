#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import GetModelState, GetModelStateRequest

rospy.init_node('odom_pub')

odom_pub = rospy.Publisher('odom', Odometry, queue_size=10)

rospy.wait_for_service('/gazebo/get_model_state')
get_model_srv = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

odom = Odometry()
model = GetModelStateRequest()
model.model_name = 'R2'  # Aquí coloca el nombre de tu modelo

r = rospy.Rate(10)  # 10hz
while not rospy.is_shutdown():
    result = get_model_srv(model)
    odom.pose.pose = result.pose
    odom.twist.twist = result.twist
    odom_pub.publish(odom)
    
    # Print position and orientation
    rospy.loginfo("Position: {}".format(result.pose.position))
    rospy.loginfo("Orientation: {}".format(result.pose.orientation))
    
    r.sleep()

