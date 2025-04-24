python main.py --data_folder pset/ \
    --problems "advantage-alignment" "Diff-Transformer" "DiffusionDPO" "DyT" "eomt" "fractalgen" "GMFlow" "GPS" "grid-cell-conformal-isometry" "hyla" "LEN" "llm-sci-use" "minp" "OptimalSteps" "REPA-E" "schedule_free" "semanticist" "SISS" "TabDiff" "Tanh-Init" \
    --gen \
    --output_dir outputs/20llms_greedy/ \
    --llm_types GEMINI_2_0_FLASH GPT_4O_2024_08_06 GPT_4O_MINI O3_MINI_HIGH DEEPSEEK_R1 O1_HIGH GPT_4_1 GPT_4_1_MINI GPT_4_1_NANO O3_HIGH CLAUDE_3_5_SONNET_2024_10_22 CLAUDE_3_7_SONNET_2025_02_19 GROK_3_BETA GEMINI_2_5_PRO_PREVIEW_03_25 \
    --n_completions 1 \
    --temperature 0 \
    --summarize_results \
    --max_retries 100 \
    --test \
    # --overwrite_test_by_llm DEEPSEEK_R1

# GEMINI_2_0_FLASH 2
# GPT_4O_2024_08_06 16.791885000000004
