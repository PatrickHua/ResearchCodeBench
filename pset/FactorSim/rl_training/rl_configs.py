import os
from ray import tune
from ray.rllib.policy import policy


# def custom_metrics_fn(info):
#     episode = info["episode"]
#     env_name = episode.env.env_name  # Retrieve the environment name from the custom attribute
#     episode.custom_metrics["env_name"] = env_name
#     return episode.custom_metrics

def custom_metrics_fn(info):
    episode = info["episode"]
    return episode.custom_metrics


def get_experiment_config(args, default_config, wandb_run_name=None):
    params_dict = {
        # resources
        "num_rollout_workers": args.num_workers,
        "num_gpus": args.num_gpus,

        # Env
        "env_name": "my_env",
        "env_config": {"name": args.env_name,
                       "method": args.method,
                       "version": args.version,
                       "max_episode_steps": args.max_episode_steps},

        # training
        # parameter explanation https://github.com/ray-project/ray/blob/c3a9756bf0c7691679edb679f666ae39614ba7e8/rllib/algorithms/algorithm_config.py#L1509        
        "seed": args.seed,
        "rollout_fragment_length": 1000,
        "train_batch_size": 10000,
        "sgd_minibatch_size": 2048,
        "disable_observation_precprocessing": True,
        "use_new_rl_modules": False,
        "use_new_learner_api": True if args.algo != 'dqn' else False,
        "framework": args.framework,

        # agent model
        "fcnet_hidden": (4, 4),
        "post_fcnet_hidden": (16,),
        "cnn_activation": "relu",
        "fcnet_activation": "relu",
        "post_fcnet_activation": "relu",
        "use_lstm": True,
        "lstm_use_prev_action": True,
        "lstm_use_prev_reward": False,
        "lstm_cell_size": 64,
        "shared_policy": False,

        # experiment trials
        "exp_name": args.exp,
        "stopping": {
                    # 10M steps
                    "timesteps_total": args.timesteps_total, 
                    # "timesteps_total": 10,
                    #"training_iteration": 1000,
                    #"episode_reward_mean": 100,
        },
        # checkpoint config
        "num_checkpoints": None,
        "checkpoint_interval": 40,
        "checkpoint_at_end": True,
        "checkpoint_score_attribute": "episode_reward_mean",
        "checkpoint_score_order": "max",
        # 
        "results_dir": args.results_dir,
        "logging": args.logging,
    }
    # tunable parameters for ray.tune.Tuner
    # default_config contains default parameter for the RL algorithm (e.g. PPO)    
    run_configs = default_config
    """ Parameter tuning:  check what's inside the default_config to 
    identify what you want to tune """
    # run_configs['gamma'] = tune.uniform(0.9, 0.999)
    # run_configs['lr'] = tune.uniform(0.0001, 0.0005)
    # run_configs['entropy_coeff'] = tune.uniform(0.01, 0.3)
    ## this is config for the tuner
    # tune_configs = tune.TuneConfig(
    #   metric='episode_reward_mean',
    #   mode='max',
    #   num_samples=200,    # number of trials, -1 means infinite
    #   reuse_actors=False)
    tune_configs = None
    """ End Parameter tuning """
    # Resources 
    run_configs.num_rollout_workers = params_dict['num_rollout_workers']
    run_configs.num_gpus = params_dict['num_gpus']

    # Training
    if args.algo != 'dqn' and args.algo != 'dreamerv3':
        run_configs.train_batch_size = params_dict['train_batch_size']
    run_configs.sgd_minibatch_size = params_dict['sgd_minibatch_size']
    run_configs.preprocessor_pref = None
    run_configs._disable_preprocessor_api = params_dict['disable_observation_precprocessing']
    # run_configs.experimental(_enable_rl_module_api=params_dict['use_new_rl_modules'])
    run_configs.experimental(_enable_new_api_stack=params_dict['use_new_learner_api'])
    run_configs = run_configs.framework(params_dict['framework'])
    run_configs.log_level = params_dict['logging']
    run_configs.seed = params_dict['seed']

    # Environment
    run_configs.env = params_dict['env_name']
    run_configs.env_config = params_dict['env_config']
    # Episode env name for meta rl
    run_configs.custom_metric_fn = custom_metrics_fn
    #run_configs.env = env_creator
    run_configs.env = "my_env"

    # Agent NN Model
    if args.use_lstm:
        # Fully connect network with number of hidden layers to be used.
        run_configs.model["fcnet_hiddens"] = params_dict['fcnet_hidden']
        # Post conv fcnet with number of hidden layers to be used.
        run_configs.model["post_fcnet_hiddens"] = params_dict['post_fcnet_hidden']
        run_configs.model["conv_activation"] = params_dict['cnn_activation'] 
        run_configs.model["fcnet_activation"] = params_dict['fcnet_activation']
        run_configs.model["post_fcnet_activation"] = params_dict['post_fcnet_activation']
        run_configs.model["use_lstm"] = params_dict['use_lstm']
        run_configs.model["lstm_use_prev_action"] = params_dict['lstm_use_prev_action']
        run_configs.model["lstm_use_prev_reward"] = params_dict['lstm_use_prev_reward']
        run_configs.model["lstm_cell_size"] = params_dict['lstm_cell_size']

    """ Adding hyper-parameter to search """
    # ray air.RunConfig
    experiment_configs = {}
    experiment_configs['name'] = params_dict['exp_name']
    experiment_configs['stop'] = params_dict['stopping']
    # checkpoint config
    experiment_configs['keep'] = params_dict['num_checkpoints']
    experiment_configs['freq'] = params_dict['checkpoint_interval']
    experiment_configs['checkpoint_score_attr'] = params_dict['checkpoint_score_attribute']
    experiment_configs['checkpoint_score_order'] = params_dict['checkpoint_score_order']
    experiment_configs['end'] = params_dict['checkpoint_at_end']

    if wandb_run_name is not None:
        experiment_configs['dir'] = f"{params_dict['results_dir']}/torch-{wandb_run_name}"
    else:
        experiment_configs['dir'] = f"{params_dict['results_dir']}/{args.framework}-{args.env_name}-{args.algo}-{args.version}"

    return run_configs, experiment_configs, tune_configs
