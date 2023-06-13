#!/usr/bin/env python

import numpy as np
import rospy
import r2env
from geometry_msgs.msg import Twist

from gym import spaces
from gym.envs.registration import register


from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, GetModelStateRequest, SetModelState

timestep_limit_per_episode = 500

register(
        id='R2Env-v0',
        entry_point='taskenv:R2TaskEnv',
        max_episode_steps=timestep_limit_per_episode,
    )

class R2TaskEnv(r2env.R2Env):
    #| Función de Inicio
    def __init__(self):

        # Definir espacio de acciones
        self.action_space = spaces.Discrete(7)

        # Definir espacio de observación
        min_distance = 0.15
        max_distance = 12.0
        num_lidar_measurements = 281
        low = np.full((num_lidar_measurements,), min_distance)
        high = np.full((num_lidar_measurements,), max_distance)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Show Info
        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

        # Iniciar contador de steps
        self.cumulated_steps = 0.0

        super(R2TaskEnv, self).__init__()


    #| Establece la pose inicial del robot
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            state_msg = ModelState()
            state_msg.model_name = 'R2'  # Asegúrate de que 'R2' es el nombre correcto del modelo
            state_msg.pose.position.x = 0.0
            state_msg.pose.position.y = 0.0
            state_msg.pose.position.z = 0.0
            state_msg.pose.orientation.x = 0.0
            state_msg.pose.orientation.y = 0.0
            state_msg.pose.orientation.z = 0.0
            state_msg.pose.orientation.w = 0.0
            state_msg.reference_frame = 'world'
            resp = set_state(state_msg)
            
            if resp.success:  # check if the call was successful
                rospy.loginfo("Model successfully set to initial pose")
            else:
                rospy.logerr("Failed to set the model to initial pose")
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)

        return True

    
    #| Variables que se inicializarán al comenzar un episodio
    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For Info Purposes
        self.cumulated_reward = 0.0
        # Set to false Done, because its calculated asyncronously
        self._episode_done = False

    #| Publica las velocidades según la acción elegida
    def _set_action(self, action):
        self.action_pub.publish(action) # El publicador se hereda de robotenv

    #| Obtener observación
    def _get_obs(self):
        rospy.logdebug("Start Get Observation ==>")
        # We get the laser scan data
        laser_scan = self.get_laser_scan()
        # We replace the 'inf' values with -1
        laser_scan = self._replace_inf_with_minus_one(laser_scan)
        rospy.logdebug("END Get Observation ==>")
        return laser_scan
    
    #| Revisar si el episodio terminó
    def _is_done(self, *args):
        return self._episode_done
    
    #| Calcular recompensa
    def _compute_reward(self, observations, done):
        reward = 1
        return reward
    

    
#! Funciones Extra
    def _replace_inf_with_minus_one(self, laser_scan):
        # Convert the laser scan data into a NumPy array
        laser_scan_np = np.array(laser_scan.ranges)
        # Replace 'inf' values with -1
        laser_scan_np[np.isinf(laser_scan_np)] = -1
        # Convert the NumPy array back into a list and return
        return laser_scan_np.tolist()