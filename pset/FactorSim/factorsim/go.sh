#!/bin/bash -ex

### set python environment
#source ~/miniconda3/etc/profile.d/conda.sh
#conda activate base

### put your api key under the home directory
# set the openapi_key
export OPENAI_API_KEY=$(cat ~/api.key)

game=$1
model="gpt-4-1106-preview" # or "gpt-3.5-turbo-1106"

python main.py $game $model
