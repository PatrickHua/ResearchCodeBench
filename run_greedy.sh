python main.py --data_folder pset/ \
    --problems "advantage-alignment" "Diff-Transformer" "DiffusionDPO" "DyT" "eomt" "fractalgen" "GMFlow" "GPS" "grid-cell-conformal-isometry" "hyla" "LEN" "llm-sci-use" "minp" "OptimalSteps" "REPA-E" "schedule_free" "semanticist" "SISS" "TabDiff" "Tanh-Init" \
    --output_dir outputs/20llms_greedy/ \
    --llm_types GEMINI_2_0_FLASH \
        GPT_4O_2024_08_06 \
        GPT_4O_MINI \
        O3_MINI_HIGH \
        DEEPSEEK_R1 \
        O1_HIGH \
        GPT_4_1 \
        GPT_4_1_MINI \
        GPT_4_1_NANO \
        O3_HIGH \
        CLAUDE_3_5_SONNET_2024_10_22 \
        CLAUDE_3_7_SONNET_2025_02_19 \
        GROK_3_BETA \
        GEMINI_2_5_PRO_PREVIEW_03_25 \
        OPENROUTER_DEEPSEEK_CHAT_V3_0324 \
        QWEN_2_5_CODER_32B_INSTRUCT \
        LLAMA_3_3_70B_INSTRUCT \
        OPENROUTER_CLAUDE_3_7_SONNET_THINKING \
        OPENROUTER_O4_MINI_HIGH \
        GROK_3_MINI_BETA_HIGH \
        GROK_2_1212 \
        OPENROUTER_CLAUDE_3_5_HAIKU \
        OPENROUTER_LLAMA_4_MAVERICK \
        OPENROUTER_LLAMA_4_SCOUT \
        OPENROUTER_GEMINI_2_5_FLASH_PREVIEW_THINKING \
    --n_completions 1 \
    --temperature 0 \
    --summarize_results \
    --max_retries 100 \
    --gen \
    --test \
    # --overwrite_test_by_prob REPA-E \
    # --overwrite_test_by_llm GEMINI_2_0_FLASH \
    # --overwrite_test_by_prob REPA-E \
    # --overwrite_gen_by_llm GPT_4O_2024_08_06 \
#  GPT_4O_2024_08_06 GPT_4O_MINI O3_MINI_HIGH DEEPSEEK_R1 O1_HIGH GPT_4_1 GPT_4_1_MINI GPT_4_1_NANO O3_HIGH CLAUDE_3_5_SONNET_2024_10_22 CLAUDE_3_7_SONNET_2025_02_19 GROK_3_BETA GEMINI_2_5_PRO_PREVIEW_03_25

        # OPENROUTER_LLAMA_4_MAVERICK \
        # OPENROUTER_LLAMA_4_SCOUT \
        # OPENROUTER_INCEPTION_MERCURY_CODER_SMALL_BETA  # length issue 32k context length
        # GEMINI_2_5_FLASH_PREVIEW_04_17
        # 
        # OPENROUTER_DEEPSEEK_CODER_V2
        # 2+
        # 
        # OPENROUTER_NVIDIA_LLAMA_3_1_NEMOTRON_ULTRA_253B_V1_FREE

        # GPT_4_5_PREVIEW \ too expensive
# GEMINI_2_0_FLASH 2
# GPT_4O_2024_08_06 16.791885000000004
