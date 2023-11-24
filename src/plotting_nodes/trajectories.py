#!/usr/bin/env python

import rospy
from gazebo_msgs.srv import GetModelState
from std_msgs.msg import Bool
import matplotlib.pyplot as plt
import time

# Configuración inicial para Matplotlib
fig, ax = plt.subplots()
ax.set_xlim([-9, 9])
ax.set_ylim([-9, 9])
ax.set_title('Trayectoria del Robot en Gazebo')
line, = ax.plot([], [], 'ro', markersize=4)  # Línea roja para la trayectoria
ax.grid(True)

trajectory_x = []
trajectory_y = []

def reset_callback(data):
    global trajectory_x, trajectory_y
    if data.data:  # Si se recibe una señal de reset
        trajectory_x = []  # Limpiar la trayectoria
        trajectory_y = []

def update_plot():
    global trajectory_x, trajectory_y

    # Esperar a que el servicio esté disponible
    rospy.wait_for_service('gazebo/get_model_state')
    try:
        get_model_state = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
        
        response = get_model_state('R2', '')
        
        x = response.pose.position.x
        y = response.pose.position.y
        trajectory_x.append(x)
        trajectory_y.append(y)

        # Actualizar la trayectoria en la gráfica
        line.set_xdata(trajectory_x)
        line.set_ydata(trajectory_y)
        plt.draw()
        plt.pause(0.01)
    
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)

def main():
    rospy.init_node('plotter_node', anonymous=True)
    rospy.Subscriber('/environment_reset', Bool, reset_callback)  # Actualizar el nombre del tópico
    
    while not rospy.is_shutdown():
        update_plot()
        time.sleep(1)
    plt.show(block=True)

if __name__ == '__main__':
    main()
