#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import curses

def main(stdscr):
    # Configuración de Curses
    stdscr.nodelay(True)  
    stdscr.timeout(100)  # Tiempo de espera del getch en milisegundos

    # Configuración de ROS
    rospy.init_node('keyboard_cmd_vel')
    pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
    rate = rospy.Rate(10)  # 10Hz
    twist = Twist()

    # Mapeo de teclas a velocidades lineales y angulares
    key_mapping = {'a': (0.8, 0.0), 's': (0.8, 0.6), 'd': (0.6, -0.8), 'f': (1.0, 0.0),
                   'z': (0.5, 0.0), 'x': (0.5, 0.4), 'c': (0.5, -0.4)}

    while not rospy.is_shutdown():
        c = stdscr.getch()

        if c != -1:  # Si se presionó una tecla
            c = chr(c)
            if c in key_mapping:
                twist.linear.x, twist.angular.z = key_mapping[c]  # Asignar velocidades correspondientes
            # Si la tecla presionada no está en el mapeo, se mantiene la última velocidad
        # Si no se presionó ninguna tecla, se mantiene la última velocidad

        pub.publish(twist)  # Publicar velocidad
        rate.sleep()

if __name__ == '__main__':
    try:
        curses.wrapper(main)
    except rospy.ROSInterruptException:
        pass
