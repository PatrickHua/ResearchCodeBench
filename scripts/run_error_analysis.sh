python -m core.analysis.generate_categories \
    --resume_from_ckpt_dir outputs/20llms_greedy/2025-04-24-13-55-17/ \
    --data_folder pset/ \
    --problems "advantage-alignment" "Diff-Transformer" "DiffusionDPO" "DyT" "eomt" "fractalgen" "GMFlow" "GPS" "grid-cell-conformal-isometry" "hyla" "LEN" "llm-sci-use" "minp" "OptimalSteps" "REPA-E" "schedule_free" "semanticist" "SISS" "TabDiff" "Tanh-Init" \
    --output_dir outputs/20llms_greedy_error_analysis/ \
    --llm_types GEMINI_2_0_FLASH GPT_4O_2024_08_06 GPT_4O_MINI O3_MINI_HIGH DEEPSEEK_R1 O1_HIGH GPT_4_1 GPT_4_1_MINI GPT_4_1_NANO O3_HIGH CLAUDE_3_5_SONNET_2024_10_22 CLAUDE_3_7_SONNET_2025_02_19 GROK_3_BETA GEMINI_2_5_PRO_PREVIEW_03_25 \
