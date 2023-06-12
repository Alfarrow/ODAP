#!/usr/bin/env python

import rospy
import gym
import numpy as np
import taskenv

def main():
    rospy.init_node('random_agent_node', anonymous=True, log_level=rospy.INFO)

    # Cargar el entorno de OpenAI que acabas de definir
    env = gym.make('R2Env-v0')

    # Número de episodios que quieres que el agente realice
    num_episodes = 100

    for episode in range(num_episodes):
        # Reiniciar el entorno al inicio de cada episodio
        obs = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # El agente toma una acción aleatoria
            # action = env.action_space.sample()
            action = 0
            
            # Aplicar la acción y obtener la respuesta del entorno
            next_obs, reward, done, info = env.step(action)
            
            total_reward += reward
            obs = next_obs

        rospy.loginfo(f"Episodio: {episode + 1}, Recompensa total: {total_reward}")

    rospy.loginfo("El agente ha completado todos los episodios")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
