import argparse


def get_cli_args():
    parser = argparse.ArgumentParser(description="Training Script for Multi-Agent RL in Meltingpot")
    parser.add_argument(
          "--env_name",
          default="default",
          help="",
    )
    parser.add_argument(
          "--env_id",
          default="default",
          help="",
    )
    parser.add_argument(
          "--exp_id",
          default="default",
          help="",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers to use for sample collection. Setting it zero will use same worker for collection and model training.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to run on (can be a fraction)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="If enabled, init ray in local mode. Tips: use this for debugging.",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="If enabled, no hyper-parameter tuning.",
    )
    parser.add_argument(
          "--method",
          choices=["ours", "baseline"],
          default="ours",
          help="env genration method",
    )
    parser.add_argument(
          "--algo",
          choices=["dreamerv3", "ppo", "dqn", "impala", "icm"],
          default="ppo",
          help="Algorithm to train agents.",
    )
    parser.add_argument(
          "--framework",
          choices=["tf2", "torch"],
          default="torch",
          help="The DL framework specifier (tf2 eager is not supported).",
    )
    parser.add_argument(
        "--exp",
        type=str,
        default="meta-exp",
        help="Name of the substrate to run",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed to run",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=10000,
        help="",
    )
    parser.add_argument(
        "--timesteps_total",
        type=int,
        default=10000000,
        help="",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="path to save results",
    )
    parser.add_argument(
          "--logging",
          choices=["DEBUG", "INFO", "WARN", "ERROR"],
          default="INFO",
          help="The level of training and data flow messages to print.",
    )
    parser.add_argument(
          "--wandb",
          action="store_true",
          default=False,
          help="Whether to use WanDB logging.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v4_6",
        help="path to save results",
    )
    ######################################### FOR EVAL
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="",
    )
    parser.add_argument(
        "--best_checkpoint_path",
        type=str,
        default=None,
        help="",
    )
    parser.add_argument(
          "--normalize",
          action="store_true",
          default=False,
          help="",
    )
    parser.add_argument(
          "--display",
          action="store_true",
          default=False,
          help="",
    )
    parser.add_argument(
          "--use_lstm",
          action="store_true",
          default=False,
          help="",
    )
    parser.add_argument(
          "--train_on_ple",
          action="store_true",
          default=False,
          help="",
    )
    parser.add_argument(
          "--downsample",
          type=bool,
          default=False,
          help="",
    )
    parser.add_argument(
          "--debug_mode",
          action="store_true",
          default=False,
          help="",
    )
    parser.add_argument(
          "--as-test",
          action="store_true",
          help="Whether this script should be run as a test.",
    )

    args = parser.parse_args()
    print("Running trails with the following arguments: ", args)
    if not args.display:
        import os
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    return args
