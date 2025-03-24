import json
import os
import numpy as np
import ray
from ray.rllib.algorithms import ppo, dqn, impala
from ray.tune import registry
from ray.rllib.utils import check_env
import pickle
from arguments import get_cli_args
from meta_rl_configs import get_experiment_config
from prepare_ple_env import PLEPygameEnv
from meta_rl_trainer import MyCustomCallback


def eval_one_episode(env, policy=None, use_lstm=False, random_policy=False):
    # Evaluate the policy
    obs, info = env.reset()
    if policy:
        state = policy.get_initial_state()
    action = 0
    done = False
    total_reward = 0
    while not done:
        if random_policy:
            action = env.action_space.sample()
            ### pong
            #if obs[0] < obs[3]:
            #    action = 1
            #else:
            #    action = 0
        else:
            if use_lstm:
                action, state, _ = policy.compute_single_action(obs, state=state, prev_action=action)
            else:
                action, _, _ = policy.compute_single_action(obs)
        obs, reward, done, _, _ = env.step(action)
        #assert abs(reward) <= 1
        total_reward += reward
        # env.render()  # This depends on the environment having a render method
    return total_reward


def ple_env_creator(config):
    env = PLEPygameEnv(config=config)
    return env


if __name__ == '__main__':
    NUM_TRIALS = 100
    args = get_cli_args()

    config={"env_config": {"name": args.env_name}} 
    registry.register_env("my_env", lambda config: ple_env_creator(config))

    if args.algo == "ppo":
        trainer = "PPO"
        from ray.rllib.algorithms import ppo
        default_config = ppo.PPOConfig()
        configs, exp_config, tune_config = get_experiment_config(args, default_config)

    elif args.algo == 'dqn':
        trainer = "DQN"
        from ray.rllib.algorithms import dqn
        default_config = dqn.DQNConfig()
        configs, exp_config, tune_config = get_experiment_config(args, default_config)

    elif args.algo == "impala":
        trainer = "IMPALA"
        from ray.rllib.algorithms import impala
        default_config = impala.ImpalaConfig()
        configs, exp_config, tune_config = get_experiment_config(args, default_config)

    elif args.algo == "icm":
       assert False

    trainer = configs.build()
    ### load a checkpoint
    if args.best_checkpoint_path:
        best_checkpoint_path = args.best_checkpoint_path
        trainer.restore(best_checkpoint_path)
        print('======================= restored ============================')
    elif args.checkpoint:
        with open(args.checkpoint, 'rb') as f:
            best_result = pickle.load(f)
        best_checkpoint_path = best_result.checkpoint.path
        trainer.restore(best_checkpoint_path)
        print('======================= restored ============================')
    else:
        pass

    policy = trainer.get_policy()

    random_rewards = []
    total_rewards = []
    for trial_idx in range(NUM_TRIALS):
        ### test on the old env
        if args.debug_mode:
            from randomize_env import env_creator
            try:
                env = env_creator({"name": args.env_name,
                                   "version": args.version,
                                   "idx": trial_idx,
                                   "method": "ours",
                                   "normalize": False,
                                   "max_episode_steps": args.max_episode_steps})
            except Exception as e:
                print(e)
                break
        else:
            env = PLEPygameEnv(config={"name": args.env_name}) # return an env instance

        random_reward = 0
        random_reward = eval_one_episode(env, random_policy=True)
        env.reset()

        total_reward = eval_one_episode(env, policy, use_lstm=args.use_lstm)
        print(f"random_reward: {random_reward:.2f}, total_reward: {total_reward:.2f}")
        random_rewards.append(random_reward)
        total_rewards.append(total_reward)

    env.close()
    ray.shutdown()
    # get the parent directory path to best_checkpoint_path
    print(random_rewards)
    print(total_rewards)
    print('###############################################')
    print('random', np.mean(random_rewards), np.std(random_rewards))
    print('total ', np.mean(total_rewards), np.std(total_rewards))
