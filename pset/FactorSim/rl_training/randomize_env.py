import importlib.util
import os
import pygame
import numpy as np
import signal
import sys
from glob import glob
from tqdm import tqdm
from env_design.wrapped_envs.pomdp_gym import PygameEnv
from env_design.wrapped_envs.oop_gym import PygameOOPEnv


def env_creator(env_config):
    env_list = []
    env_name = env_config["name"]

    if env_config["method"] == "ours":
        version, numbered_directory = env_config['version'].split('_')
        directory_path = f"env_design/generated_envs/{env_name}_eval/{version}/"
        for filename in os.listdir(directory_path):
            # Load the module from the file
            if filename.startswith('pomdp'):
                # Construct a full file path
                file_path = directory_path + '/' + filename + f'/{numbered_directory}/' + 'final.py'
                # Define module name based on file name and load the Game module from the file
                module_name = filename
                try:
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    env_list.append((module_name, getattr(module, f"Game")))
                except Exception as e:
                    print(f"Fail to load {module_name}")

        env_config["env_list"] = sorted(env_list)
        env = PygameEnv(env_config)
        return env

    elif env_config["method"] == "baseline":
        version  = env_config['version']
        directory_path = f"env_design/generated_envs/{env_name}_baseline_eval/{version}"
        for filename in os.listdir(directory_path):
            # Load the module from the file
            if filename.endswith(".py"):
                # Construct a full file path
                file_path = directory_path + '/' + filename
                # Define module name based on file name and load the Game module from the file
                module_name = filename
                try:
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    env_list.append((module_name, getattr(module, f"Game")))
                except Exception as e:
                    print(f"Fail to load {module_name}")

        env_config["env_list"] = sorted(env_list)
        env = PygameOOPEnv(env_config)
        return env
            
    else:
        assert False

def get_env_list(env_config):
    env_list = []
    env_name = env_config["name"]

    if env_config["method"] == "ours":
        version, numbered_directory = env_config['version'].split('_')
        directory_path = f"env_design/generated_envs/{env_name}_eval/{version}/"
        for filename in os.listdir(directory_path):
            # Load the module from the file
            if filename.startswith('pomdp'):
                # Construct a full file path
                file_path = directory_path + '/' + filename + f'/{numbered_directory}/' + 'final.py'
                # Define module name based on file name and load the Game module from the file
                module_name = filename
                try:
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    env_list.append((module_name, getattr(module, f"Game")))
                except Exception as e:
                    print(f"Fail to load {module_name}")

        env_config["env_list"] = sorted(env_list)
        return env_config["env_list"]
        #env = PygameEnv(env_config)
        #return env

    elif env_config["method"] == "baseline":
        version  = env_config['version']
        directory_path = f"env_design/generated_envs/{env_name}_baseline_eval/{version}"
        for filename in os.listdir(directory_path):
            # Load the module from the file
            if filename.endswith(".py"):
                # Construct a full file path
                file_path = directory_path + '/' + filename
                # Define module name based on file name and load the Game module from the file
                module_name = filename
                try:
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    env_list.append((module_name, getattr(module, f"Game")))
                except Exception as e:
                    print(f"Fail to load {module_name}")

        env_config["env_list"] = sorted(env_list)
        return env_config["env_list"]
        #env = PygameOOPEnv(env_config)
        #return env
            
    else:
        assert False


def my_env_check(env):
    total_rewards = []
    for _ in range(5):
        print('start evaluating ...')
        try:
            observation, _ = env.reset()
            done = False
            total_reward = 0
            idx = 0
            for i in tqdm(range(500)):
                action = env.action_space.sample()
                observation, reward, done, truncated, info = env.step(action)
                if info["has_exception"]:
                    return False
                total_reward += reward
                if done:
                    break
            total_rewards.append(total_reward)
            print(f"total reward: {total_reward}")
            pygame.quit()
            env.close()
        except Exception as e:
            print(e)
            return False

    print(total_rewards)
    if env.env_name == "pong":
        # check if all the values are the same
        if np.all([total_rewards[i] == total_rewards[0] for i in range(len(total_rewards))]):
            return False
        if np.sum(total_rewards) == 0:
            return False
    return True
