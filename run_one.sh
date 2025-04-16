
python main.py --data_folder pset/ \
    --problems "Diff-Transformer" \
    --gen \
    --output_dir outputs/eval_harper_dev/ \
    --llm_types GEMINI_2_0_FLASH \
    --n_completions 1 \
    --temperature 0.8 \
    --summarize_results \
    --max_retries 100 \
    --test