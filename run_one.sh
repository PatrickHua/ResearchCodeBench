

# O1_HIGH
python main.py --data_folder pset/ \
    --problems "hyla" "DyT" "DiffusionDPO" "LEN" "fractalgen" "llm-sci-use" "minp" "eomt" \
    --gen \
    --test \
    --output_dir outputs/debug/baseline_o1_high_enhanced_prompt/ \
    --llm_types O1_HIGH \
    --n_completions 1 \
    --temperature 0.8 \
    --max_iter 10 \
    --summarize_results \


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

