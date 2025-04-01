
problem_name="llm-sci-use"
python main.py --src_folder pset/ \
    --run_one $problem_name \
    --gen \
    --test \
    --llm_types "gemini-1.5-flash" \
    --n_completions 1 \
    --temperature 0.6 \
    --max_iter 10 \
    --summarize_results \
