#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections


#| Neural Network
class Net(nn.Module):
    def __init__(self, num_actions):
        super(Net, self).__init__()
        self.ConvBranch = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=16, kernel_size=9, stride=4),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=33*32 , out_features=100),
            nn.ReLU()
        )
        self.LinearBranch = nn.Sequential(
            nn.Linear(181, 181),
            nn.ReLU(),
            nn.Linear(181, 100),
            nn.ReLU()
        )
        self.joined_layer = nn.Sequential(
            nn.Linear(400, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
        
    def forward(self, inputLidar, inputOrientation):
        # Pasar entradas por sus respectivos branches
        outputLidar = self.ConvBranch(inputLidar)
        outputOrientation = self.LinearBranch(inputOrientation)
        outputOrientation = outputOrientation.view(outputOrientation.shape[0], -1) 

        # Pasar por capa densa
        joined_output = torch.cat((outputLidar, outputOrientation), dim=1)
        final_output = self.joined_layer(joined_output)
        
        return final_output
    
#| Clases y funciones que servirán al agente en su entrenamiento
#* Clase Experience replay: Almacena obs, acción, recompensa, fin, siguiente_obs
class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) # Cola de longitud máxima capacity
        
    def __len__(self):
        return len(self.buffer)
    
    def append(self, experience):
        self.buffer.append(experience)
        
    def sample(self, batch_size):                       # Selecciona una muestra aleatoria de tamaño batch_size
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        observations, actions, rewards, dones, next_observations = zip(*[self.buffer[idx] for idx in indices])
        return observations, np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), next_observations

#| Tupla "Experience" que se almacenará en ExperienceReplay 
Experience = collections.namedtuple('Experience', field_names=['obs', 'action', 'reward', 'done', 'new_obs'])

#| Clase Agente
class DQN_Agent:
    def __init__(self, env, exp_replay):
        self.env = env
        self.exp_replay = exp_replay
        self._reset()
        
    def _reset(self):
        self.obs_actual = self.env.reset()
        self.recompensa_total = 0.0
        
    def step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None                                  #! Aquí se guardará el retorno de todo el episodio
        if np.random.random() < epsilon:
            accion = self.env.action_space.sample()         #! Acción aleatoria según la política epsilon-greedy
        else:                                               #! Acción óptima según la política epsilon-greedy
            obsLidar_, obsOri_ = adjust_input(self.obs_actual)       
            obsLidar = torch.from_numpy(obsLidar_).float().unsqueeze(0).to(device)
            obsOri = torch.from_numpy(obsOri_).float().unsqueeze(0).to(device)
            q_vals = net(obsLidar, obsOri)                              # Procesar el obs con la red neuronal para
                                                                        # obtener los valores Q
            _, acc_ = torch.max(q_vals, dim=1)                          # _ = valor máximo, acc_ = índice de la acción
            accion = int(acc_.item())                                   # Convertir el índice de la acción a un entero
            
        nuevo_obs, recompensa, is_done, info = self.env.step(accion)          #! Ejecutar la acción en el entorno
        self.recompensa_total += recompensa
        # print(nuevo_obs[5])
        # print(recompensa)
        # print("Total: ", self.recompensa_total)
        
        exp = Experience(self.obs_actual, accion, recompensa, is_done, nuevo_obs) # Almacenar la experiencia en una tupla
        self.exp_replay.append(exp)                           #! Añadir la tupla al buffer ExperienceReplay
        self.obs_actual = nuevo_obs
        
        if is_done:
            done_reward = self.recompensa_total
            self._reset()
        return done_reward
    
#| Funciones de procesamiento de la observación
#* Función de normalización
def normalize_laser_readings(readings, max_distance):
    return np.array(readings) / max_distance

#* Función para codificación 
def encode_orientation(orientation):
    # Convert orientation from radians to degrees and take absolute value
    orientation_degrees = abs(np.degrees(orientation))

    # Map orientation from [0, 180] to [0, 179]
    index = int(orientation_degrees // 2)

    # Create one-hot encoded vector
    one_hot = np.zeros(181)  # 180 for orientation + 1 for sign
    one_hot[index] = 1

    # Set last element to 1 if orientation was positive, 0 if it was negative
    one_hot[-1] = 1 if orientation >= 0 else 0

    return one_hot

#* Función para ajustar entradas para la red neuronal
def adjust_input(obs, max_distance=12.0):
    # Normalizar escaneos láser, cada escaneo es de 281
    las1 = normalize_laser_readings(obs[0], max_distance)
    las2 = normalize_laser_readings(obs[1], max_distance)
    las3 = normalize_laser_readings(obs[2], max_distance)

    # Convertir ángulo a vector de 181 elementos donde el último indica si es negativo o positivo
    ang1 = encode_orientation(obs[3][1])
    ang2 = encode_orientation(obs[4][1])
    ang3 = encode_orientation(obs[5][1])

    # Apilar las medidas láser para formar la entrada de la rama convolucional
    lidar_input = np.stack((las1, las2, las3))

    # Apilar las orientaciones para formar la entrada de la rama lineal
    orientation_input = np.stack((ang1, ang2, ang3))

    return lidar_input, orientation_input


