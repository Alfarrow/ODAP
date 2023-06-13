#!/usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from std_msgs.msg import Int32
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState, ContactsState
from gazebo_msgs.srv import GetModelState, GetModelStateRequest, SetModelState
from openai_ros import robot_gazebo_env


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


        # Revisar que todo esté listo
        self._check_all_systems_ready()

    
    #| Revisar que todos los sistemas estén listos (sensores, publicadores, ...)
    def _check_all_systems_ready(self):
        self._check_laser_scan_ready()
        self._check_bumper_ready()
        self._check_odom_ready()
        self._check_publishers_connection()

        return True
    
    #| Callback del LiDAR
    def _laser_scan_callback(self, data):
        self.laser_scan = data

    #| Función para obtener los datos guardados del lidar
    def get_laser_scan(self):
        return self.laser_scan
    
    #| Callback de Bumper
    def _bumper_callback(self, contact_data):
        """
        Callback function for bumper topic subscriber.
        """
        if contact_data.states:  # If the bumper sensor returns a non-empty list, the robot hit something
            self._episode_done = True
    
#! Funciones vacías que se configurarán de preferencia en el Task Environment
    # def _set_init_pose(self):
    #     """Sets the Robot in its init pose
    #     """
    #     raise NotImplementedError()
    
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

        




