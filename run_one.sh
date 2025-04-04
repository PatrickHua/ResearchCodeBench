

python main.py --src_folder pset/ \
    --problems "hyla" "DyT" "DiffusionDPO" "LEN" "fractalgen" "llm-sci-use" \
    --gen \
    --test \
    --llm_types "gemini-1.5-flash" "gemini-2.0-flash" "gpt-4o-mini" "gpt-4o" \
    --n_completions 1 \
    --temperature 0.8 \
    --max_iter 10 \
    --summarize_results