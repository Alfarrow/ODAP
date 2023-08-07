#!/usr/bin/env python

import rospy
import collections
import numpy as np
from rospy import get_param
from geometry_msgs.msg import Twist

import r2env_v1
from gym import spaces
from gym.envs.registration import register

from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, GetModelStateRequest, SetModelState



register(
        id='R2Env-v1',
        entry_point='r2_multi_goal_env_v1:R2TaskEnv',
        max_episode_steps=get_param('Training/timestep_limit_per_episode')
    )

class R2TaskEnv(r2env_v1.R2Env):
    #| Función de Inicio
    def __init__(self):

        # Nombre del robot
        self.robot_name = get_param("Training/robot_name")

        # Definir espacio de acciones
        self.action_space = spaces.Discrete(7)

        # Definir espacio de observación
        min_distance = get_param('Training/min_distance')
        max_distance = get_param('Training/max_distance')
        self.num_lidar_measurements = get_param('Training/num_lidar_measurements')
        low = np.full((self.num_lidar_measurements,), min_distance)
        high = np.full((self.num_lidar_measurements,), max_distance)
        laser_box = spaces.Box(low, high, dtype=np.float32)
        self.observation_space = spaces.Tuple([
            laser_box,
            laser_box,
            laser_box,
            spaces.Box(low=np.array([0.0, -np.pi]), high=np.array([np.inf, np.pi]), dtype=np.float32)
        ])

        # Show Info
        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

        # Iniciar contador de steps
        self.cumulated_steps = 0.0
        self.max_steps_reached = False
        self.max_episode_steps = get_param('Training/timestep_limit_per_episode')

        # Cargar configuración de entrenamiento
        #* Meta
        self.goals_axis_x = get_param('Training/goals_axis_x')
        self.goals_axis_y = get_param('Training/goals_axis_y')
        self.goal_index = 0
        self.goal_x = self.goals_axis_x[self.goal_index]
        self.goal_y = self.goals_axis_y[self.goal_index]
        self.threshold_goal = get_param('Training/threshold_goal')
        print("Primer Objetivo: ", self.goal_x, self.goal_y)

        # Variables que se usarán para el cálculo de la recompensa
        self.last_dist_to_goal = self.calculate_dist_to_goal()
        self.last_angle_to_goal = self.calculate_angle_to_goal()


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
            state_msg.pose.orientation.w = 1.0
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
        # Vaciar los buffers
        self.laser_scan_buffer.clear()
        self.distance_buffer.clear()
        self.orientation_buffer.clear()
        # For Info Purposes
        self.cumulated_steps = 0.0
        # Set to false Done, because its calculated asyncronously
        self._episode_done = False
        # Establecer velocidad en 0's
        self.action_pub.publish(7)
        # Reiniciar metas
        self.goal_index = 0
        self.goal_x = self.goals_axis_x[self.goal_index]
        self.goal_y = self.goals_axis_y[self.goal_index]

        self.last_dist_to_goal = self.calculate_dist_to_goal()
        self.last_angle_to_goal = self.calculate_angle_to_goal()
        self.collision = False

        # Valores por defecto para los bufferes
        default_value_laser_scan = np.full((self.num_lidar_measurements,), -1.0) # reemplazar num_lidar_measurements con la dimensión de tus datos láser
        self.laser_scan_buffer.append(default_value_laser_scan.tolist())  # Asegúrate de convertir la matriz numpy a lista con .tolist()

        self.distance_buffer.append(self.last_dist_to_goal)
        self.orientation_buffer.append(self.last_angle_to_goal)

    #| Publica las velocidades según la acción elegida
    def _set_action(self, action):
        self.cumulated_steps += 1  # Incrementa el número de pasos realizados
        if self.cumulated_steps >= self.max_episode_steps:  # Si se han alcanzado los pasos máximos
            self.max_steps_reached = True  # Marca la variable como True
            self._episode_done = True  # Y también indica que el episodio ha terminado
        self.action_pub.publish(action)

    #| Obtener observación
    def _get_obs(self):
        # Lista para almacenar los últimos escaneos láser
        last_scans = list(self.laser_scan_buffer)
        distances = list(self.distance_buffer)
        angles = list(self.orientation_buffer)

        # Verificar y llenar los buffers si es necesario (esto sólo se hace la primera vez creo)
        if len(last_scans) < self.laser_scan_buffer.maxlen:
            last_scans += [last_scans[-1]] * (self.laser_scan_buffer.maxlen - len(last_scans))
        
        if len(distances) < self.distance_buffer.maxlen:
            distances += [distances[-1]] * (self.distance_buffer.maxlen - len(distances))

        if len(angles) < self.orientation_buffer.maxlen:
            angles += [angles[-1]] * (self.orientation_buffer.maxlen - len(angles))


        return [last_scans[0], last_scans[1], last_scans[2],
                np.array([distances[0], angles[0]]), np.array([distances[1], angles[1]]), np.array([distances[2], angles[2]])]
   
    #| Revisar si el episodio terminó
    def _is_done(self, *args):
        distance_to_goal = self.calculate_dist_to_goal()

        # Revisar si se llegó al objetivo
        # if distances[-1] <= self.threshold_goal: # Al igual que en las recompensas esto no corresponde a la realidad
        if distance_to_goal <= self.threshold_goal:
            self.goal_index += 1
            if self.goal_index >= len(self.goals_axis_x):  # Si se alcanzó el último objetivo
                self._episode_done = True
            else:  # Si todavía hay más objetivos por alcanzar
                self.goal_x = self.goals_axis_x[self.goal_index]
                self.goal_y = self.goals_axis_y[self.goal_index]
                print("Nuevo objetivo: ", self.goal_x, self.goal_y)

        return self._episode_done
    
    #| Calcular recompensa
    def _compute_reward(self, observations, done):
        # Obtenerlo de la realidad
        current_dist = self.calculate_dist_to_goal()
        current_angle = self.calculate_angle_to_goal()
        last_dist_to_goal = self.last_dist_to_goal
        last_angle_to_goal = self.last_angle_to_goal
        self.last_dist_to_goal = current_dist  # Actualiza la última distancia
        self.last_angle_to_goal = current_angle

        reward = 0

        # Recompensa o castigo por doblar o no
        if self.latest_linear_velocity != 0 and self.latest_angular_velocity == 0:
            reward += 0.01  # Recompensa si hay velocidad lineal pero no angular

        if self.latest_angular_velocity != 0:
            reward -= 0.01  # Castigo si hay velocidad angular

        # Castigo si la observación del lidar indica un objeto a menos de 1 metros
        positive_observations = [obs for obs in observations[2] if obs > 0]
        if positive_observations:
            min_distance = min(positive_observations)
            if min_distance < 1.0:
                reward -= (1.0 - min_distance) * 0.01

        if done:
            # Velocidades en 0's
            self.action_pub.publish(7) # 7 es la acción que coloca en ceros las velocidades
            
            # Recompensa cada vez que el agente se orienta mejor hacia el objetivo
            reward += (abs(last_angle_to_goal) - abs(current_angle)) * 10

            if self.collision:  # Si hubo una colisión
                reward -= 20
            elif current_dist <= self.threshold_goal:  # Si se está a 20 cm o menos de la meta
                if self.goal_index >= len(self.goals_axis_x):  # Si es el último objetivo
                    reward += 20
                    print("Meta Final Alcanzada: +20")
                else:
                    pass
            else:
                reward += 0  # Este caso no debería suceder, pero se deja por seguridad

        else:
            # Recompensa cada vez que el agente se orienta mejor hacia el objetivo
            reward += (abs(last_angle_to_goal) - abs(current_angle)) * 10

            # Si se alcanzo uno de los objetivos (no final)
            if current_dist <= self.threshold_goal:
                if self.goal_index >= len(self.goals_axis_x):
                    pass
                else:
                    reward += 20
                    print("Un objetivo alcanzado: +20")

        return reward


    
  

