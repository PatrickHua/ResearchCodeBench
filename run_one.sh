

python main.py --src_folder pset/ \
    --problems "hyla" "DyT" "DiffusionDPO" "LEN" "fractalgen" "llm-sci-use" \
    --gen \
    --test \
    --llm_types "gemini-2.0-flash" "gpt-4o" "o3-mini-2025-01-31" "deepseek-reasoner" "claude-3-5-sonnet-20241022" \
    --n_completions 1 \
    --temperature 0.8 \
    --max_iter 10 \
    --summarize_results