
python main.py --data_folder pset/ \
    --problems REPA-E \
    --gen \
    --output_dir outputs/eval_harper_dev/REPA-E2/ \
    --llm_types GPT_4O GEMINI_2_5_FLASH_PREVIEW_04_17 GEMINI_2_5_PRO_PREVIEW_03_25 O3_HIGH \
    --n_completions 1 \
    --temperature 0.8 \
    --summarize_results \
    --max_retries 100 \
    --overwrite_test_by_llm GPT_4O \
    --overwrite_test REPA-E \
    --test