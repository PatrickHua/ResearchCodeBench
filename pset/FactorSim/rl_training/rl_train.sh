#!/bin/bash -ex

### Assume the environments are generated already
# export OPENAI_API_KEY=

#### find your wandb api key here: https://wandb.ai/authorize
export WANDB_API_KEY=
export TUNE_RESULT_DIR=$(pwd)/results

### meta rl experiments
env_name=$1
algo=$2
exp_id=$3
results_dir=$TUNE_RESULT_DIR
echo "logging at $results_dir"

wandb_run_name=torch-$env_name-$version-$exp_id

#python randomize_env.py $env_name $version
TMPDIR=
python rl_trainer.py --env_name $env_name --algo $algo --exp_id $exp_id --results_dir $results_dir  --wandb --num_gpus 1 --num_workers 40 "${@:4}"
python rl_eval.py    --env_name $env_name --algo $algo --exp_id $exp_id --results_dir $results_dir --checkpoint $results_dir/$wandb_run_name/best_result.pkl "${@:4}"
