
python main.py --data_folder pset/ \
    --problems "DiffusionDPO" "DyT" "eomt" "fractalgen" "GMFlow" "hyla" "LEN" "llm-sci-use" "minp" "OptimalSteps" "schedule_free" "semanticist" \
    --gen \
    --output_dir outputs/12llms_wo_paper/ \
    --llm_types GEMINI_2_0_FLASH GPT_4O_MINI GPT_4O_2024_08_06 O3_MINI_HIGH DEEPSEEK_R1 O1_HIGH \
    --n_completions 5 \
    --temperature 0.8 \
    --summarize_results \
    --max_retries 100 \
    --test \
    --wo_paper

    #  "eomt" "fractalgen" "GMFlow" "hyla" "LEN" "llm-sci-use" "minp" "OptimalSteps" "schedule_free" "semanticist" 