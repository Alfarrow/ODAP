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
        
        # Capas de convolución 1D
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=5, stride=3)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2)
        
        # Capa densa
        self.fc1 = nn.Linear(in_features=32*76, out_features=256)  # Ajustar el tamaño de entrada según sea necesario
        
        # Capa de salida
        self.fc2 = nn.Linear(in_features=256, out_features=num_actions)  # Ajustar el tamaño de salida según sea necesario

    def forward(self, x):
        # Pasar la entrada a través de las capas de convolución con ReLU como función de activación
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Aplanar la salida de las capas de convolución
        x = x.view(x.size(0), -1)
        
        # Pasar la salida aplanada a través de la capa densa con ReLU como función de activación
        x = F.relu(self.fc1(x))
        
        # Pasar la salida de la capa densa a través de la capa de salida
        # No se aplica ninguna función de activación aquí ya que esto depende de tu problema
        # Por ejemplo, si estás haciendo una clasificación multiclase, podrías usar una función Softmax aquí
        x = self.fc2(x)
        
        return x
    
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
            obs_ = adjust_input(self.obs_actual)       
            obs = torch.from_numpy(obs_).float().unsqueeze(0).to(device)
            q_vals = net(obs)                                        # Procesar el obs con la red neuronal para
                                                                        # obtener los valores Q
            _, acc_ = torch.max(q_vals, dim=1)                          # _ = valor máximo, acc_ = índice de la acción
            accion = int(acc_.item())                                   # Convertir el índice de la acción a un entero
            
        nuevo_obs, recompensa, is_done, info = self.env.step(accion)          #! Ejecutar la acción en el entorno
        self.recompensa_total += recompensa
        
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

    # Concatenar las observaciones para cada canal
    channel1 = np.concatenate((las1, ang1))
    channel2 = np.concatenate((las2, ang2))
    channel3 = np.concatenate((las3, ang3))

    # Apilar los canales para formar la entrada completa
    input_array = np.stack((channel1, channel2, channel3))

    return input_array

