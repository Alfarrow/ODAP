#!/usr/bin/env python

import gym
import rospy
from std_msgs.msg import Float32
import torch
import r2_multi_goal_env_v0
from DQN_v0 import Net, DQN_Agent
from DQN_v0 import adjust_input

def main():
    rospy.init_node('DQN_Agent', anonymous=True, log_level=rospy.INFO)

    #| Crear redes neuronales, entorno y agente
    env = gym.make('R2Env-v1')

    # Redes neuronales
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net(env.action_space.n).to(device)
    #* Pesos guardados antes
    checkpoint_net = torch.load('/home/alfarrow/trained_models/1st_training_orientation/Final_Orientation.pth')
    net.load_state_dict(checkpoint_net)
    print("<Checkpoints Net Cargados>")


    rospy.loginfo("|------|Evaluación Comenzada|-----|")

    while True:
        done = False
        obs = env.reset()
        total_reward = 0

        while not done:
            # Procesar Observación
            obsLidar, obsOri = adjust_input(obs)
            obsLidar = torch.from_numpy(obsLidar).float().unsqueeze(0).to(device)
            obsOri = torch.from_numpy(obsOri).float().unsqueeze(0).to(device)
            # Tomar decisión
            q_vals = net(obsLidar, obsOri)
            _, act = torch.max(q_vals, dim=1)
            action = int(act.item())

            next_obs, reward, done, _ = env.step(action)

            total_reward += reward
            obs = next_obs

        rospy.loginfo(f"Recompensa del episodio: {total_reward}")
    
        # Podrías querer detener la evaluación después de un cierto número de episodios
        # break 

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
