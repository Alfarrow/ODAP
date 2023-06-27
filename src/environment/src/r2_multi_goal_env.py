#!/usr/bin/env python

import rospy
import collections
import numpy as np
from rospy import get_param
from geometry_msgs.msg import Twist

import r2env
from gym import spaces
from gym.envs.registration import register

from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, GetModelStateRequest, SetModelState

from tf.transformations import euler_from_quaternion


register(
        id='R2Env-v1',
        entry_point='r2_multi_goal_env:R2TaskEnv',
        max_episode_steps=get_param('Training/timestep_limit_per_episode')
    )

class R2TaskEnv(r2env.R2Env):
    #| Función de Inicio
    def __init__(self):

        # Nombre del robot
        self.robot_name = get_param("Training/robot_name")

        # Definir espacio de acciones
        self.action_space = spaces.Discrete(7)

        # Definir espacio de observación
        min_distance = get_param('Training/min_distance')
        max_distance = get_param('Training/max_distance')
        num_lidar_measurements = get_param('Training/num_lidar_measurements')
        low = np.full((num_lidar_measurements,), min_distance)
        high = np.full((num_lidar_measurements,), max_distance)
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

    #| Publica las velocidades según la acción elegida
    def _set_action(self, action):
        self.cumulated_steps += 1  # Incrementa el número de pasos realizados
        if self.cumulated_steps >= self.max_episode_steps:  # Si se han alcanzado los pasos máximos
            self.max_steps_reached = True  # Marca la variable como True
            self._episode_done = True  # Y también indica que el episodio ha terminado
        self.action_pub.publish(action)

    #| Obtener observación
    def _get_obs(self):
        rospy.logdebug("Start Get Observation ==>")

        # Lista para almacenar los últimos escaneos láser
        last_scans = []
        # Lista para distancias y ángulos al objetivo
        distances = []
        angles = []

        # Repetir si no se tienen todos los escaneos que se requieren
        while len(last_scans) < 3:
            # We get the laser scan data
            scan = self.get_laser_scan()

            # Reemplazar 'inf' con -1
            scan = self._replace_inf_with_minus_one(scan)

            # Añadir a la lista
            last_scans.append(scan)

            # Añadir distancia y ángulo al objetivo
            dist_to_goal = self.calculate_dist_to_goal()
            angle_to_goal = self.calculate_angle_to_goal()
            distances.append(dist_to_goal)
            angles.append(angle_to_goal)

        # Revisar si se llegó al objetivo
        if distances[2] <= self.threshold_goal:
            self.goal_index += 1
            if self.goal_index >= len(self.goals_axis_x):  # Si se alcanzó el último objetivo
                self._episode_done = True
            else:  # Si todavía hay más objetivos por alcanzar
                self.goal_x = self.goals_axis_x[self.goal_index]
                self.goal_y = self.goals_axis_y[self.goal_index]
                print("Nuevo objetivo: ", self.goal_x, self.goal_y)

        rospy.logdebug("END Get Observation ==>")


        if len(last_scans) == 3:
            return [last_scans[0], last_scans[1], last_scans[2],
                    np.array([distances[0], angles[0]]), np.array([distances[1], angles[1]]), np.array([distances[2], angles[2]])]
        else:
            rospy.logwarn("Not enough scans in last_scans!")
            return None
    
    #| Revisar si el episodio terminó
    def _is_done(self, *args):
        
        return self._episode_done
    
    #| Calcular recompensa
    def _compute_reward(self, observations, done):
        current_dist = observations[5][0]
        current_angle = observations[5][1]
        last_dist_to_goal = self.last_dist_to_goal
        last_angle_to_goal = self.last_angle_to_goal
        self.last_dist_to_goal = current_dist  # Actualiza la última distancia
        self.last_angle_to_goal = current_angle

        reward = 0

        if done:
            # Recompensa mayor a medida que el agente se acerca más al objetivo
            reward += (last_dist_to_goal - current_dist) * 100

            # Recompensa cada vez que el agente se orienta mejor hacia el objetivo
            reward += (abs(last_angle_to_goal) - abs(current_angle)) * 100

            if self.collision:  # Si hubo una colisión
                reward -= 20
            elif self.max_steps_reached:  # Si se alcanzó el número máximo de pasos
                reward -= 20
            elif current_dist <= self.threshold_goal:  # Si se está a 10 cm o menos de la meta
                if self.goal_index >= len(self.goals_axis_x):  # Si es el último objetivo
                    reward += 20
                    print("Meta Final Alcanzada: +50")
                else:
                    pass
            else:
                reward += 0  # Este caso no debería suceder, pero se deja por seguridad
                print("Sí sucedió")
        else:
            # Recompensa mayor a medida que el agente se acerca más al objetivo
            reward += (last_dist_to_goal - current_dist) * 100

            # Recompensa cada vez que el agente se orienta mejor hacia el objetivo
            reward += (abs(last_angle_to_goal) - abs(current_angle)) * 100

            # Si se alcanzo uno de los objetivos (no final)
            if current_dist <= self.threshold_goal:
                if self.goal_index >= len(self.goals_axis_x):
                    pass
                else:
                    reward += 20
                    print("Un objetivo alcanzado: +20")

        return reward


    
#! Funciones Extra
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
    

