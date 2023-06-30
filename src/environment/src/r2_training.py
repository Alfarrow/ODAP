#!/usr/bin/env python

import gym
import time
import rospy
from std_msgs.msg import Float32
import torch
import torch.nn as nn
from torch import optim
import r2_multi_goal_env_v1
import numpy as np
from DQN_v0 import Net, DQN_Agent, ExperienceReplay
from DQN_v0 import encode_orientation, normalize_laser_readings, adjust_input


def main():
    # Iniciar Nodo
    rospy.init_node('DQN_Agent', anonymous=True, log_level=rospy.INFO)

    # Crear publicadores para las métricas
    episode_reward_publisher = rospy.Publisher('reward', Float32, queue_size=10)
    mean_reward_publisher = rospy.Publisher('mean_reward', Float32, queue_size=10)


    #| Cargar parámetros de entrenamiento
    # Condiciones para calificar el aprendizaje del agente
    MEAN_REWARD_BOUND = rospy.get_param("Training/mean_reward_bound") # Promedio de recompensas para considerar que el agente ha aprendido
    NUMBER_OF_REWARDS_TO_AVERAGE = rospy.get_param("Training/number_of_rewards_to_average") # Número de episodios para calcular el promedio de recompensas

    # Hiperparámetros de entrenamiento
    BATCH_SIZE = rospy.get_param("Training/batch_size")
    LEARNING_RATE = rospy.get_param("Training/learning_rate")
    GAMMA = rospy.get_param("Training/gamma")
    EPS_START = rospy.get_param("Training/eps_start")
    EPS_DECAY = rospy.get_param("Training/eps_decay")
    EPS_MIN = rospy.get_param("Training/eps_min")
    EXPERIENCE_REPLAY_SIZE = rospy.get_param("Training/exp_replay_size") # Tamaño del buffer de experiencia
    SYNC_TARGET_NETWORK = rospy.get_param("Training/sync_target_network") # Número de pasos para sincronizar la red objetivo con la red de entrenamiento

    #| Crear redes neuronales, entorno y agente
    # Entorno
    env = gym.make('R2Env-v1')

    # Redes neuronales
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net(env.action_space.n).to(device)
    target_net = Net(env.action_space.n).to(device)
    #* Pesos guardados antes
    # checkpoint_net = torch.load('/home/alfarrow/trained_models/1st_training_orientation/Final_573.pth')
    # net.load_state_dict(checkpoint_net)
    # print("<Checkpoints Net Cargados>")

    # checkpoint_target = torch.load('/home/alfarrow/trained_models/1st_training_orientation/Final_573.pth')
    # target_net.load_state_dict(checkpoint_target)
    # print("<Checkpoints Target Cargados>")

    # Iniciar agente y experience replay
    buffer = ExperienceReplay(EXPERIENCE_REPLAY_SIZE)
    agente = DQN_Agent(env, buffer)

    #| Bucle de entrenamiento
    # Iniciar parámetros de entrenamiento
    epsilon = EPS_START
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    numero_frame = 0                # Contador
  
    rospy.loginfo("|------|Entrenamiento Comenzado|-----|")

    while True:
        #* Agente obtiene experiencia
        numero_frame += 1
        epsilon = max(EPS_MIN, epsilon * EPS_DECAY)     # Actualizar Epsilon

        reward = agente.step(net, epsilon, device=device)   # Ejecutar un paso del agente

        #* Cuando se acaba el episodio
        if reward is not None:                              #! Contar recompensas si el episodio ha terminado
            total_rewards.append(reward)
            mean_reward = np.mean(total_rewards[-NUMBER_OF_REWARDS_TO_AVERAGE:])
            rospy.loginfo(f"Frame:{numero_frame} | Total de episodios:{len(total_rewards)} | Recompensa del Episodio: {total_rewards[len(total_rewards)-1]} | Recompensa Media: {mean_reward:.3f}  (Epsilon usado ={epsilon:.2f})")
            rospy.loginfo("%d:  %d session, recompensa media %.3f, (epsilon %.2f)" % (numero_frame, len(total_rewards), mean_reward, epsilon))

            # Publicar las métricas
            episode_reward_publisher.publish(Float32(total_rewards[len(total_rewards)-1]))
            mean_reward_publisher.publish(Float32(mean_reward))

            # Guardar la red cada 10 episodios
            if len(total_rewards) % 20 == 0:
                rospy.loginfo(f"Guardando el modelo en el episodio {len(total_rewards)}")
                torch.save(net.state_dict(), f"/home/alfarrow/trained_models/checkpoint_{len(total_rewards)}.pth")
                torch.save(target_net.state_dict(), f"/home/alfarrow/trained_models/checkpoint_target_{len(total_rewards)}.pth")

            
            
            if mean_reward > MEAN_REWARD_BOUND:             #! Si el promedio de recompensas es mayor que el límite terminar
                rospy.loginfo(f"RESUELTO en {numero_frame} frames y {len(total_rewards)} sessions")
                rospy.loginfo(f"Guardando el modelo en el episodio {len(total_rewards)}")
                torch.save(net.state_dict(), f"/home/alfarrow/trained_models/Final_{len(total_rewards)}.pth")
                break

        #* Entrenamiento
        if len(buffer) < BATCH_SIZE:                        #! Si el buffer no está lleno no se entrena
            continue

        batch = buffer.sample(BATCH_SIZE)                   # Obtener batch de experiencias del buffer
        observations_, actions_, rewards_, dones_, new_observations_ = batch

        #* Convertir batches a tensores para que sean compatibles con la red neuronal y se puedan enviar al GPU
        actions = torch.tensor(actions_).to(device)
        rewards = torch.tensor(rewards_).to(device)
        dones = torch.tensor(dones_, dtype=torch.bool).to(device)
        # Es ligeramente diferente para las observaciones
        observations = [adjust_input(obs) for obs in observations_]
        new_observations = [adjust_input(new_obs) for new_obs in new_observations_]

        lidar_inputs = torch.from_numpy(np.array([obs[0] for obs in observations])).float().to(device)
        orientation_inputs = torch.from_numpy(np.array([obs[1] for obs in observations])).float().to(device)

        new_lidar_inputs = torch.from_numpy(np.array([new_obs[0] for new_obs in new_observations])).float().to(device)
        new_orientation_inputs = torch.from_numpy(np.array([new_obs[1] for new_obs in new_observations])).float().to(device)



        # Calcular los valores Q de la red neuronal principal
        Q_values = net(lidar_inputs, orientation_inputs).gather(1, actions.unsqueeze(-1).long()).squeeze(-1) #! 1. Se procesan los estados
                                                                                         #! 2. Gather: Se seleccionan los valores Q de las acciones tomadas anteriormente
                                                                                         #! 3. squeeze(-1): Se elimina la dimensión de tamaño 1

        # Calcular los valores Q de la red neuronal target
        new_obs_values = target_net(new_lidar_inputs, new_orientation_inputs).max(1)[0]
        new_obs_values[dones] = 0.0   # Si el episodio ha terminado no se tiene en cuenta el valor del nuevo estado
        new_obs_values = new_obs_values.detach() # No se calculan los gradientes de los valores del nuevo estado debido a que no se van a entrenar
        
        # Calcular los valores Q esperados
        expected_Q_values = rewards + GAMMA * new_obs_values
        # Calcular el error
        loss = nn.MSELoss()(Q_values, expected_Q_values)
        
        #| MODIFICAR LOS PESOS DE LA RED NEURONAL
        optimizer.zero_grad() # Coloca los gradientes a cero
        loss.backward()       # Calcula los gradientes
        optimizer.step()      # Actualiza los pesos
        
        #| SINCRONIZAR LA RED TARGET
        if numero_frame % SYNC_TARGET_NETWORK == 0:
            target_net.load_state_dict(net.state_dict())
            



if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
