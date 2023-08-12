#!/usr/bin/env python

import rospy
import time
from std_msgs.msg import Int32
import matplotlib.pyplot as plt

# Variables globales para almacenar el total de éxitos y fallos
success_count = 0
fail_count = 0

# Configuración inicial para Matplotlib
labels = ['Éxitos', 'Fallos']
sizes = [success_count, fail_count]
colors = ['green', 'red']
explode = (0.1, 0)  # explode 1st slice for emphasis

def success_callback(data):
    global success_count
    success_count = data.data

def fail_callback(data):
    global fail_count
    fail_count = data.data

def update_pie_chart():
    sizes[0] = success_count
    sizes[1] = fail_count
    plt.clf()  # Clear the current figure
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)

    # Añadir leyenda con episodios totales
    total_episodes = success_count + fail_count
    plt.legend([f"Episodios totales: {total_episodes}"], loc="upper right")

    plt.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle
    plt.pause(0.01)


def main():
    rospy.init_node('pie_chart_node', anonymous=True)
    rospy.Subscriber('/total_success', Int32, success_callback)
    rospy.Subscriber('/total_fails', Int32, fail_callback)

    while not rospy.is_shutdown():
        update_pie_chart()
        time.sleep(1)

    plt.show(block=True)
    rospy.spin()

if __name__ == '__main__':
    main()
