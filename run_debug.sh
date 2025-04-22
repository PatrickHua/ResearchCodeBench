
python main.py --data_folder pset/ \
    --problems "DiffusionDPO" \
    --gen \
    --output_dir outputs/12llms_debug/ \
    --llm_types GPT_4O_MINI \
    --n_completions 1 \
    --temperature 0 \
    --summarize_results \
    --max_retries 100 \
    --test
