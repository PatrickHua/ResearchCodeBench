from PIL import Image
from prepare_ple_env import ple_env_creator
import imageio
import numpy as np
import os
import pygame
import sys


def save_frame(env, save_path):
    rgb_array = env.game.getScreenRGB()
    image = Image.fromarray(np.transpose(rgb_array, (1, 0, 2)))
    image.save(save_path)


def rule_based_policy(env_name, observation):
    if env_name == "snake":
        if observation[0] < observation[3]:
            return 1
        else:
            return 0
    elif env_name == "pong":
        if observation[0] < observation[1]:
            return 1
        else:
            return 0
    elif env_name == "waterworld":
        if observation[0] < observation[6]:
            return 2
        else:
            return 1
    elif env_name == "catcher":
        if observation[0] < observation[1]:
            return 2
        else:
            return 1
    else:
        raise ValueError(f"env_name {env_name} rule based policy not supported")


if __name__ == "__main__":
    #for env_name in ["snake", "pong", "waterworld", "snake"]:
    env_name = sys.argv[1]
    root_save_dir = "saved_videos/"
    NUM_TRIALS = 100
    max_episode_steps = 10000
    policy_type = "rule_based" # could be "rule_based" or "ppo"

    pygame.init()

    env_config = {"name": env_name}
    env = ple_env_creator(env_config) 
    # PLEPygameEnv(config=env_config) # return an env instance
    print(f"{env_name} passed")
    print(env.p.getActionSet())

    episode_rewards = []
    episode_lengths = []

    for trial_idx in range(NUM_TRIALS):
        frame_paths = []
        save_dir = os.path.join(root_save_dir, env_name, str(trial_idx))
        os.makedirs(save_dir, exist_ok=True)
        step_i = 0
        total_reward = 0
        done = False
        observation, _ = env.reset()
        while not done:
            if policy_type == "random":
                action = env.action_space.sample()
            elif policy_type == "rule_based":
                action = rule_based_policy(env_name, observation)
            else:
                raise ValueError(f"policy_type {policy_type} not supported")
            

            observation, reward, done, truncated, info = env.step(action)
            total_reward += reward

            ### 
            
            image_path = os.path.join(save_dir, f"{step_i}.png")
            save_frame(env, image_path)
            frame_paths.append(image_path)

            step_i += 1
            if done:
                env.reset()
            if step_i > max_episode_steps:
                break

        video_path = os.path.join(root_save_dir, env_name, f"{trial_idx}.gif")
        with imageio.get_writer(video_path, mode='I', duration=0.5) as writer:
            for frame_path in frame_paths:
                image = imageio.imread(frame_path)
                writer.append_data(image)

        episode_lengths.append(step_i)
        episode_rewards.append(total_reward)
        print(f"episdoe_length: {step_i}, total_reward: {total_reward}")

    print(episode_lengths)
    print(episode_rewards)
    print('mean episode_rewards over {NUM_TRIALS}', np.mean(episode_rewards))
    env.close()