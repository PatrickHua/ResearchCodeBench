
problem_name="hyla"
problem_name2="DyT"
problem_name3="DiffusionDPO"
problem_name4="LEN"

python main.py --src_folder pset/ \
    --problems $problem_name $problem_name2 $problem_name3 $problem_name4 \
    --gen \
    --test \
    --llm_types "gemini-1.5-flash" "gpt-4o" \
    --n_completions 1 \
    --temperature 0.6 \
    --max_iter 10 \
    --summarize_results