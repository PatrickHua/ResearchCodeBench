import numpy as np
import os
import sys
from randomize_env import env_creator
from prepare_ple_env import ple_env_creator
import matplotlib.pyplot as plt
import numpy as np


def plot(data1, data2, labels):
    # Sample data: Replace with your actual data
    # For demonstration, random data is generated with a normal distribution
    #data1 = np.random.normal(50, 10, 1000)
    #data2 = np.random.normal(60, 15, 1000)

    # Set up the 2x2 grid of subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    for i in range(len(labels)):
        ix, iy = i // 2, i % 2
        # Upper left plot
        axes[ix, iy].hist(data1[:, i], bins=30, alpha=0.5, label='generated')
        axes[ix, iy].hist(data2[:, i], bins=30, alpha=0.5, label='gt')
        axes[ix, iy].legend()
        axes[ix, iy].set_title(labels[i])

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig("env_dist.png")
    #plt.show()


def get_env_state_dist(env, episodes=5):
    states = []
    rewards = []
    total_rewards = []
    for _ in range(episodes):
        observation, _ = env.reset()
        done = False
        total_reward = 0
        idx = 0
        while not done:
            action = env.action_space.sample()
            states.append(observation)
            observation, reward, done, truncated, info = env.step(action)
            rewards.append(reward)
            total_reward += reward
            idx += 1
            if idx >= 10000:
                break
        total_rewards.append(total_reward)
        #print(f"Total reward: {total_reward}")
        env.close()

    print(total_rewards)
    all_states = np.array(states)
    rewards = np.array(rewards).reshape(-1, 1)
    print(all_states.shape)
    for i in range(all_states.shape[1]):
        print(f"state {i}, mean: {np.mean(all_states[:, i])}, std: {np.std(all_states[:, i])}, min: {np.min(all_states[:, i])}, max: {np.max(all_states[:, i])}")
    #return all_states
    # #np.concatenate([all_states, rewards], axis=1)
    return all_states, rewards


if __name__ == '__main__':
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    #game = "flappy_bird"
    #game = "catcher"
    game = sys.argv[1]
    version = sys.argv[2]
    #labels = ['player_y', 'cpu_y', 'x_dist', 'ball_y']
    if game == "flappy_bird":
        labels = ["player_y", "player_vel", "next_pipe_dist_to_player", "next_pipe_bottom_y"]

    #version = "v8_4"
    #for method in ["one_shot", "debug", "sequential", "ours"]:
    method = "ours"
    directory_path = f"env_design/generated_envs/{game}_eval/{version.split('_')[0]}"
    num_folders = len([name for name in os.listdir(directory_path) if name.startswith('pomdp')])    
    print('num_envs ', num_folders)

    failed_env_list = []
    for idx in range(num_folders):
        env_config = {"name": game, "method": method, "idx": idx, "version": version, "max_episode_steps": 10000}
        env = env_creator(env_config)
        all_states, all_rewards = get_env_state_dist(env)

        print('==================================')
        env = ple_env_creator(env_config)
        gt_states, gt_rewards = get_env_state_dist(env, episodes=20)
        # selected_env_id = env.env_list[idx][0]
        # repeat gt_states so that gt_shapes.shape[0] == all_states.shape[0]
        #gt_states = np.repeat(gt_states, all_states.shape[0] // gt_states.shape[0], axis=0)
        print(all_states.shape, gt_states.shape)

        plot(all_states, gt_states, labels)
