python main.py --data_folder pset/ \
    --problems "advantage-alignment" "Diff-Transformer" "DiffusionDPO" "DyT" "eomt" "fractalgen" "GMFlow" "GPS" "grid-cell-conformal-isometry" "hyla" "LEN" "llm-sci-use" "minp" "OptimalSteps" "REPA-E" "schedule_free" "semanticist" "SISS" "TabDiff" "Tanh-Init" \
    --gen \
    --output_dir outputs/20llms_greedy/ \
    --llm_types GEMINI_2_0_FLASH GPT_4O_2024_08_06 GPT_4O_MINI O3_MINI_HIGH \
    --n_completions 1 \
    --temperature 0 \
    --summarize_results \
    --max_retries 100 \
    --test

# GEMINI_2_0_FLASH 2
# GPT_4O_2024_08_06 16.791885000000004
