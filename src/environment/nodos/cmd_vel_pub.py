#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32
from rospy.exceptions import ROSTimeMovedBackwardsException

class VelocityPublisherNode:
    def __init__(self):
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.action_sub = rospy.Subscriber('/action', Int32, self.action_callback)
        self.vel_cmd = Twist()
        self.actions = [(0.8, 0.0), (0.8, 0.6), (0.8, -0.6), (1.0, 0.0), (0.5, 0.0), (0.5, 0.4), (0.5, -0.4), (0.0, 0.0)]

    def action_callback(self, msg):
        try:
            linear_vel, angular_vel = self.actions[msg.data]
            self.vel_cmd.linear.x = linear_vel
            self.vel_cmd.angular.z = angular_vel
        except IndexError:
            rospy.logerr("Action index out of range!")

    def run(self):
        rate = rospy.Rate(10)  # 10Hz
        while not rospy.is_shutdown():
            try:
                self.vel_pub.publish(self.vel_cmd)
                rate.sleep()
            except ROSTimeMovedBackwardsException:
                rospy.logwarn("ROS Time moved backwards. Continuing the loop...")
                linear_vel, angular_vel = self.actions[-1]  # Set to the last speed in the list
                self.vel_cmd.linear.x = linear_vel
                self.vel_cmd.angular.z = angular_vel
                continue

if __name__ == '__main__':
    try:
        rospy.init_node('velocity_publisher_node')
        node = VelocityPublisherNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.logwarn("ROS Interrupted. Exiting Velocity Publisher Node.")

