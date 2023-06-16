#!/usr/bin/env python

import rospy
import matplotlib.pyplot as plt
from std_msgs.msg import Float32
import threading
import queue

rewards = []
mean_rewards = []
q = queue.Queue()

def reward_callback(msg):
    rewards.append(msg.data)
    q.put(None)

def mean_reward_callback(msg):
    mean_rewards.append(msg.data)
    q.put(None)

def main():
    rospy.init_node('metrics_plot_node')

    reward_sub = rospy.Subscriber('/reward', Float32, reward_callback)
    mean_reward_sub = rospy.Subscriber('/mean_reward', Float32, mean_reward_callback)

    fig, ax = plt.subplots()
    plt.ion()

    while not rospy.is_shutdown():
        try:
            while not q.empty():
                q.get()
                ax.clear()
                ax.plot(rewards, label='Reward')
                ax.plot(mean_rewards, label='Mean Reward')
                ax.scatter(range(len(rewards)), rewards)
                ax.scatter(range(len(mean_rewards)), mean_rewards)
                ax.legend()
                plt.pause(0.01)

            rospy.sleep(0.1)
        except rospy.exceptions.ROSTimeMovedBackwardsException:
            pass

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()
