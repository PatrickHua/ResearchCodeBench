
python main.py --data_folder pset/ \
    --problems "DiffusionDPO" "DyT" "eomt" "fractalgen" "GMFlow" "hyla" "LEN" "llm-sci-use" "minp" "OptimalSteps" "schedule_free" "semanticist" \
    --gen \
    --test \
    --output_dir outputs/12llms/ \
    --llm_types GEMINI_2_0_FLASH GPT_4O_MINI GPT_4O_2024_08_06 O3_MINI_HIGH \
    --n_completions 5 \
    --temperature 0.8 \
    --summarize_results \
    --max_retries 100 \
    --overwrite_test semanticist

    # --problems "DiffusionDPO" "DyT" "eomt" "fractalgen" "GMFlow" "hyla" "LEN" "llm-sci-use" "minp" "OptimalSteps" "schedule_free" "semanticist" \
# 
# # O1_HIGH
# python main.py --data_folder pset/ \
#     --problems "hyla" "DyT" "DiffusionDPO" "LEN" "fractalgen" "llm-sci-use" "minp" "eomt" \
#     --gen \
#     --test \
#     --output_dir outputs/baseline_10paper/ \
#     --llm_types O1_HIGH \
#     --n_completions 1 \
#     --temperature 0.8 \
#     --max_iter 10 \
#     --summarize_results \


# python main.py --data_folder pset/ \
#     --problems "hyla" "DyT" "DiffusionDPO" "LEN" "fractalgen" "llm-sci-use" "minp" "eomt" \
#     --gen \
#     --test \
#     --output_dir outputs/debug/baseline_o3_mini_high_enhanced_prompt/ \
#     --llm_types O3_MINI_HIGH \
#     --n_completions 1 \
#     --temperature 0.8 \
#     --max_iter 10 \
#     --summarize_results \

# python main.py --data_folder pset/ \
#     --problems "hyla" "DyT" "DiffusionDPO" "LEN" "fractalgen" "llm-sci-use" "minp" "eomt" \
#     --gen \
#     --test \
#     --output_dir outputs/debug/baseline_minp_claude-3-5_enhanced_prompt/ \
#     --llm_types "claude-3-5-sonnet-20241022" \
#     --n_completions 1 \
#     --temperature 0.8 \
#     --max_iter 10 \
#     --summarize_results \

