from ray import air
from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import EpisodeType, PolicyID
from ray.tune import registry
from typing import *
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Type, Union
if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.env.env_runner import EnvRunner
    from ray.rllib.evaluation import WorkerSet
import gc
import gymnasium as gym
import json
import numpy as np
import os
import pickle
import platform
# Import psutil after ray so the packaged version is used.
# disble tensorflow warning logs
import psutil
import ray
import tracemalloc
import wandb
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from arguments import get_cli_args
from randomize_env import env_creator
from prepare_ple_env import ple_env_creator
from rl_configs import get_experiment_config


class CustomWandbLoggerCallback(WandbLoggerCallback):
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        env_name = base_env.get_unwrapped()[0].env_name
        wandb.config.update({"env_name": env_name}, allow_val_change=True)
        super().on_episode_end(worker=worker, base_env=base_env, policies=policies,
                               episode=episode, env_index=env_index, **kwargs)

class MyCustomCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()

    def on_episode_end(
        self,
        *,
        # TODO (sven): Deprecate Episode/EpisodeV2 with new API stack.
        episode: Union[EpisodeType, Episode, EpisodeV2],
        # TODO (sven): Deprecate this arg new API stack (in favor of `env_runner`).
        worker: Optional["EnvRunner"] = None,
        env_runner: Optional["EnvRunner"] = None,
        # TODO (sven): Deprecate this arg new API stack (in favor of `env`).
        base_env: Optional[BaseEnv] = None,
        env: Optional[gym.Env] = None,
        # TODO (sven): Deprecate this arg new API stack (in favor of `rl_module`).
        policies: Optional[Dict[PolicyID, Policy]] = None,
        rl_module: Optional[RLModule] = None,
        env_index: int,
        **kwargs,
    ) -> None:
        # Optionally, you can decide the frequency of evaluations
        self.alt_env = ple_env_creator({"name": args.env_name})
        assert len(policies) == 1
        policy = policies["default_policy"]
        total_reward = 0.0
        num_episodes = 5

        # Evaluate the policy on the alternative environment
        for _ in range(num_episodes):
            obs, _ = self.alt_env.reset()
            state = policy.get_initial_state()
            use_lstm = len(state.keys()) > 0
            action = 0
            done = False
            episode_reward = 0.0
            while not done:
                if use_lstm:
                    action, state, _ = policy.compute_single_action(obs, state, prev_action=action)
                else:
                    action, _, _ = policy.compute_single_action(obs)
                obs, reward, done, _, _ = self.alt_env.step(action)
                episode_reward += reward
            total_reward += episode_reward

        avg_reward = total_reward / num_episodes
        # Add the average reward from the alternative environment to the result dict
        episode.custom_metrics["alt_env_avg_reward"] = avg_reward


if __name__ == "__main__":
    global args
    args = get_cli_args()
    # Set up Ray. Use local mode for debugging. Ignore reinit error.
    # Register environment

    if args.train_on_ple:
        from prepare_ple_env import ple_env_creator
        registry.register_env("my_env", lambda config: ple_env_creator(config))
    else:
        registry.register_env("my_env", lambda config: env_creator(config))

    # Setup WanDB    
    if "WANDB_API_KEY" in os.environ and args.wandb:
      wandb_project = f'{args.exp}_{args.framework}'
      wandb_group = f'{args.env_name}-{args.algo}'
      wandb_run_name = f'{args.env_name}-{args.version}-{args.exp_id}' + \
                        ('-train_on_ple' if args.train_on_ple else '')

      # Set up Weights And Biases logging if API key is set in environment variable.
      wdb_callbacks = [
            CustomWandbLoggerCallback(
                project=wandb_project,
                name=wandb_run_name,
                group=wandb_group,
                api_key=os.environ["WANDB_API_KEY"],
                log_config=True,
          ),
      ]
    else:
      wdb_callbacks = []
      print("WARNING! No wandb API key found, running without wandb!")

    # initialize default configurations for native RLlib algorithms (we use one solver 
    # all exploration modules)  
    # Fetch experiment configurations
    #from configs import get_experiment_config
    # hyperparameter search
    if args.algo == "dreamerv3":
        assert False
        #from ray.rllib.algorithms import dreamerv3
        #trainer = "DreamerV3"
        #num_gpus = args.num_gpus
        #default_config = dreamerv3.DreamerV3Config()
        #configs, exp_config, tune_config = get_experiment_config(args, default_config ,wandb_run_name=wandb_run_name)

    if args.algo == "ppo":
        trainer = "PPO"
        from ray.rllib.algorithms import ppo
        default_config = ppo.PPOConfig().callbacks(MyCustomCallback)
        configs, exp_config, tune_config = get_experiment_config(args, default_config, wandb_run_name=wandb_run_name)

    elif args.algo == 'dqn':
        trainer = "DQN"
        from ray.rllib.algorithms import dqn
        default_config = dqn.DQNConfig()
        default_config["callbacks"] = MyCustomCallback
        configs, exp_config, tune_config = get_experiment_config(args, default_config, wandb_run_name=wandb_run_name)


    elif args.algo == "impala":
        trainer = "IMPALA"
        from ray.rllib.algorithms import impala
        default_config = impala.ImpalaConfig()
        default_config["callbacks"] = MyCustomCallback
        configs, exp_config, tune_config = get_experiment_config(args, default_config, wandb_run_name=wandb_run_name)

    elif args.algo == "icm":
       assert False

    else:
        print('The selected option is not tested. You may encounter issues if you use the baseline \
              policy configurations with non-tested algorithms')

    # Ensure GPU is available if set to True
    #if configs.num_gpus > 0 and args.framework == "torch":
    #   import torch
    #   if torch.cuda.is_available():
    #      print("Using GPU device.")
    #   else:
    #      print("Either GPU is not available on this machine or not visible to this run. Training using CPU only.")
    #      configs.num_gpus = 0

    ### I changed the _temp_dir but you likely don't have to do so
    # _temp_dir='',
    ray.init(local_mode=args.local, ignore_reinit_error=True)
    ### for some reason, I have to set the following line, otherwiseI will get "URI has empty scheme"
    # exp_config['dir'] = None

    # Setup hyper-parameter optimization configs here
    # hyperparam tuning - documentation for tune.TuneConfig: https://docs.ray.io/en/latest/tune/api/doc/ray.tune.TuneConfig.html#ray.tune.TuneConfig
    if not args.no_tune:
     # NotImplementedError
     tune_config = None
    else:
     tune_config = tune.TuneConfig(reuse_actors=False)

    # Setup checkpointing configurations documentation
    # https://docs.ray.io/en/latest/train/api/doc/ray.train.CheckpointConfig.html?highlight=checkpoint_score_attribute
    ckpt_config = air.CheckpointConfig(
      num_to_keep=exp_config['keep'],
      checkpoint_score_attribute=exp_config['checkpoint_score_attr'],
      checkpoint_score_order=exp_config['checkpoint_score_order'],
      checkpoint_frequency=exp_config['freq'],
      checkpoint_at_end=exp_config['end'])

    # documentation for air.RunConfig https://github.com/ray-project/ray/blob/c3a9756bf0c7691679edb679f666ae39614ba7e8/python/ray/air/config.py#L575
    run_config=air.RunConfig(
      name=exp_config['name'],
      callbacks=wdb_callbacks,
      storage_path=exp_config['dir'],
      local_dir=exp_config['dir'],
      stop=exp_config['stop'],
      checkpoint_config=ckpt_config,
      verbose=3)

    # Run Trials documentation https://docs.ray.io/en/latest/tune/api/doc/ray.tune.Tuner.html#ray-tune-tuner  
    results = tune.Tuner(
        trainer, # trainable to be tuned
        param_space=configs.to_dict(),
        tune_config=tune_config,
        run_config=run_config,
    ).fit()

    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
    print(best_result)

    # save best_result to a pickle file
    with open(f"{exp_config['dir']}/best_result.pkl", "wb") as f:
        pickle.dump(best_result, f)
    # load the best checkpoint
    #with open(f"{exp_config['dir']}/best_result.pkl", "rb") as f:
    #    best_result = pickle.load(f)
