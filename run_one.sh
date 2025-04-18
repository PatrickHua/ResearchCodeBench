
python main.py --data_folder pset/ \
    --problems REPA-E \
    --gen \
    --output_dir outputs/eval_harper_dev/REPA-E3/ \
    --llm_types O3_MINI_HIGH \
    --n_completions 1 \
    --temperature 0.8 \
    --summarize_results \
    --max_retries 100 \
    --test