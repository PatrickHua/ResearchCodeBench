
problem_name="eomt"
python main.py --src_folder pset/ \
    --run_one $problem_name \
    --gen \
    --test \
    --llm_types "gpt-4o-mini" "o3-mini" \
    --n_completions 2 \
    --temperature 0.6 \
    --max_iter 10 \
    --summarize_results \
    --overwrite