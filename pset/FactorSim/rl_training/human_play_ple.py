import numpy as np
import sys
import numpy as np
from prepare_ple_env import ple_env_creator
import pygame


if __name__ == "__main__":
    #for env_name in ["snake", "pong", "waterworld", "snake"]:
    NUM_TRIALS = 10
    max_episode_steps = 10000
    game_name = sys.argv[1]
    pygame.init()

    for env_name in [game_name]:
        env_config = {"name": env_name}
        env = ple_env_creator(env_config) 
        # PLEPygameEnv(config=env_config) # return an env instance
        print(f"{env_name} passed")
        print(env.p.getActionSet())

        episode_rewards = []
        episode_lengths = []
        for _ in range(NUM_TRIALS):
            i = 0
            total_reward = 0
            done = False
            observation, _ = env.reset()
            while not done:
                action = env.action_space.sample()
                ### pong
                #if observation[0] < observation[3]:
                #    action = 1
                #else:
                #    action = 0

                ### puckworld
                #if observation[0] < observation[6]:
                #    action = 2
                #else: 
                #    action = 1
                #elif observation[1] < observation[5]:
                #    action = 4

                ### catcher
                #if observation[0] < observation[1]:
                #    action = 2
                #else:
                #    action = 1

                ### waterworld
                observation, reward, done, truncated, info = env.step(action)
                total_reward += reward
                i += 1
                if done:
                    env.reset()
                #if i > max_episode_steps:
                #    break
            episode_lengths.append(i)
            episode_rewards.append(total_reward)
            print(total_reward)

        print(episode_lengths)
        print(episode_rewards)
        print('mean episode_rewards over {NUM_TRIALS}', np.mean(episode_rewards))
        env.close()
