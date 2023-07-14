#!/usr/bin/env python

import numpy as np
from collections import deque
import rospy
from sensor_msgs.msg import LaserScan, Imu
from std_msgs.msg import Bool
from std_msgs.msg import Int32
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState, ContactsState
from gazebo_msgs.srv import GetModelState, GetModelStateRequest, SetModelState
from openai_ros import robot_gazebo_env

from tf.transformations import euler_from_quaternion


class R2Env(robot_gazebo_env.RobotGazeboEnv):
#! Funciones Necesarias  
    #| Función de inicio
    def __init__(self):

        # Especificar controladores
        self.controllers_list = []
        
        # Namespace
        self.robot_name_space = "R2Env"

        # Se reiniciarán los controladores?
        reset_controls_bool = False

        # Se lanza la función de inicialización de la clase padre 'robot_gazebo_env.RobotGazeboEnv'
        super(R2Env, self).__init__(controllers_list=self.controllers_list,
                                                robot_name_space=self.robot_name_space,
                                                reset_controls=reset_controls_bool)
        
        # Crear publicador
        self.action_pub = rospy.Publisher('/action', Int32, queue_size=1)

        # Crear suscriptores
        rospy.Subscriber("/R2/scan", LaserScan, self._laser_scan_callback)
        rospy.Subscriber("/R2/bumper", ContactsState, self._bumper_callback)
        rospy.Subscriber("/R2/imu", Imu, self._imu_callback)
        self.velocity_subscriber = rospy.Subscriber("/cmd_vel", Twist, self.velocity_callback)
        self.latest_linear_velocity = 0
        self.latest_angular_velocity = 0

        # Variable para detectar si se está en colisión
        self.collision = False

        # Inicializar buffers
        self.laser_scan_buffer = deque(maxlen=3)
        self.distance_buffer = deque(maxlen=3)
        self.orientation_buffer = deque(maxlen=3)

        # Revisar que todo esté listo
        self._check_all_systems_ready()

    
    #| Revisar que todos los sistemas estén listos (sensores, publicadores, ...)
    def _check_all_systems_ready(self):
        self._check_laser_scan_ready()
        self._check_imu_ready()
        self._check_bumper_ready()
        self._check_odom_ready()
        self._check_publishers_connection()

        return True
    
    #| Callback del LiDAR
    def _laser_scan_callback(self, data):
        self.laser_scan = data
        laser_scan_processed = self._replace_inf_with_minus_one(data)
        self.laser_scan_buffer.append(laser_scan_processed)

        # Agregar distancia y orientación a sus respectivos bufferes
        distance_to_goal = self.calculate_dist_to_goal()
        orientation_to_goal = self.calculate_angle_to_goal_imu()
        self.distance_buffer.append(distance_to_goal)
        self.orientation_buffer.append(orientation_to_goal)

    #| Función para obtener los datos guardados del lidar
    def get_laser_scan(self):
        return self.laser_scan
    
    #| Callback del sensor IMU
    def _imu_callback(self, data):
        self.imu_data = data
    
    #| Callback de Bumper
    def _bumper_callback(self, contact_data):
        """
        Callback function for bumper topic subscriber.
        """
        if contact_data.states:  # If the bumper sensor returns a non-empty list, the robot hit something
            self._episode_done = True
            self.collision = True

    #| Callback Velocidad
    # Función callback para el suscriptor de velocidad
    def velocity_callback(self, data):
        self.latest_linear_velocity = data.linear.x
        self.latest_angular_velocity = data.angular.z
            
    
#! Funciones vacías que se configurarán de preferencia en el Task Environment
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()
    
    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()
    
    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()


#! Funciones que permitirán obtener datos para la observación
    # Reemplazar inf con -1
    def _replace_inf_with_minus_one(self, laser_scan):
        # Convert the laser scan data into a NumPy array
        laser_scan_np = np.array(laser_scan.ranges)
        # Replace 'inf' values with -1
        laser_scan_np[np.isinf(laser_scan_np)] = -1
        # Convert the NumPy array back into a list and return
        return laser_scan_np.tolist()
    
    # Obtener posición del robot
    def get_robot_pose(self):
        model_state_client = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        try:
            response = model_state_client(self.robot_name, 'world')
            return response.pose.position
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)
            return None

    # Obtener orientación del robot
    def get_robot_orientation(self):
        model_state_client = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        try:
            response = model_state_client(self.robot_name, 'world')
            orientation_q = response.pose.orientation
            orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
            (_, _, yaw) = euler_from_quaternion(orientation_list)
            return yaw
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)
            return None

    # Calcular distancia al objetivo
    def calculate_dist_to_goal(self):
        robot_pose = self.get_robot_pose()

        # Distancia
        dist_to_goal = np.sqrt((self.goal_x - robot_pose.x)**2 + (self.goal_y - robot_pose.y)**2)

        return dist_to_goal
       
    # Calcular ángulo al objetivo
    def calculate_angle_to_goal(self):
        robot_pose = self.get_robot_pose()

        # Ángulo global al objetivo
        global_angle_to_goal = np.arctan2(self.goal_y - robot_pose.y, self.goal_x - robot_pose.x)

        # Orientación del robot
        robot_orientation = self.get_robot_orientation()

        # Ángulo relativo al objetivo
        relative_angle_to_goal = global_angle_to_goal - robot_orientation

        # Normaliza el ángulo relativo al objetivo para asegurarte de que siempre esté en el rango [-pi, pi]
        relative_angle_to_goal = (relative_angle_to_goal + np.pi) % (2 * np.pi) - np.pi

        return relative_angle_to_goal
    
    # Calcular ángulo al objetivo usando IMU
    def calculate_angle_to_goal_imu(self):
        robot_pose = self.get_robot_pose()

        # Obtener orientación del robot a partir de los datos del IMU
        orientation_q = self.imu_data.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (_, _, yaw) = euler_from_quaternion(orientation_list)

        # Ángulo global al objetivo
        global_angle_to_goal = np.arctan2(self.goal_y - robot_pose.y, self.goal_x - robot_pose.x)

        # Ángulo relativo al objetivo
        relative_angle_to_goal = global_angle_to_goal - yaw

        # Normaliza el ángulo relativo al objetivo para asegurarte de que siempre esté en el rango [-pi, pi]
        relative_angle_to_goal = (relative_angle_to_goal + np.pi) % (2 * np.pi) - np.pi

        return relative_angle_to_goal

#! Funciones Extra
    # Revisar disponibilidad de LiDAR
    def _check_laser_scan_ready(self):
        self.laser_scan = None

        while self.laser_scan is None and not rospy.is_shutdown():
            try:
                self.laser_scan = rospy.wait_for_message("/R2/scan", LaserScan, timeout=1.0)
                rospy.logdebug("Current /scan READY=>")

            except:
                rospy.logerr("Current /R2/scan not ready yet, retrying for getting laser_scan")
        return self.laser_scan

    # Revisar disponibilidad del sensor IMU 
    def _check_imu_ready(self):
        self.imu_data = None
        while self.imu_data is None and not rospy.is_shutdown():
            try:
                self.imu_data = rospy.wait_for_message("/R2/imu", Imu, timeout=1.0)
                rospy.logdebug("Current /R2/imu READY=>")
            except:
                rospy.logerr("Current /R2/imu not ready yet, retrying for getting imu")
        return self.imu_data
    
    # Revisar sensor Bumper
    def _check_bumper_ready(self):
        self.bumper = None

        while self.bumper is None and not rospy.is_shutdown():
            try:
                self.bumper = rospy.wait_for_message("/R2/bumper", ContactsState, timeout=1.0)  # Corregido aquí
                rospy.logdebug("Current /R2/bumper READY=>")

            except:
                rospy.logerr("Current /R2/bumper not ready yet, retrying for getting bumper state")
        return self.bumper
    
    # Revisar que se esté publicando la localización del robot
    def _check_odom_ready(self):
        rospy.wait_for_service('/gazebo/get_model_state')
        get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

        try:
            resp = get_model_state(model_name='R2')
        except rospy.ServiceException:
            return False

        return True
    

    # Revisar que los publicadores funcionen
    def _check_publishers_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rate = rospy.Rate(10)  # 10hz
        while self.action_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No susbribers to action_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("action_pub Publisher Connected")

        rospy.logdebug("All Publishers READY")

        




