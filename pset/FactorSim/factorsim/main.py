import os
import sys
from evaluation.evaluate_utils import execute_file, dynamical_import, run_eval
from utils import save_json_to_file, add_initial_states
from factorized_pomdp import GameRep
from prompts import high_level_decompose_prompt


if __name__ == "__main__":
    # Set these game constants
    game_name = sys.argv[1]
    model = sys.argv[2]

    USE_DECOMPOSE_PROMPT = True
    NUM_TRIALS = 10
    WIDTH, HEIGHT, FPS = 1000, 1000, 30
    # make the directory if it does not exist
    game_path = f"./games/single_player_games/{game_name}"
    implementation_path = os.path.join(game_path, "mdp.py")
    test_file_path = os.path.join(game_path, "mdp_unit_test.py")

    all_prompts, backbone_code = dynamical_import(game_path)

    save_dir = f"./factorsim_results/{model}/{game_name}_eval"
    os.makedirs(save_dir, exist_ok=True)
    for idx in range(NUM_TRIALS):
        log_dir = f"{save_dir}/pomdp_{idx}"
        os.makedirs(log_dir, exist_ok=True)
        implementation_path = log_dir + "/final.py"
        game = GameRep(
            HEIGHT,
            WIDTH,
            FPS,
            MAX_RETRIES=3,
            log_dir=log_dir,
            debug_mode=1,
            model=model,
        )
        ## add default States to the game
        add_initial_states(game)
        ## To train RL agents, we need to initialize the states needed for the (downstream) RL agent

        ### PROMPT 1: get step by step plan
        if USE_DECOMPOSE_PROMPT:
            for _ in range(game.MAX_RETRIES):
                steps = game.ask_llm(
                    high_level_decompose_prompt.format(
                        game_specification=all_prompts
                    )
                )
                try:
                    steps = game.ask_llm(
                        high_level_decompose_prompt.format(
                            game_specification=all_prompts
                        )
                    )
                    assert "steps" in steps
                    save_json_to_file(steps, f"{log_dir}/decompose_{idx}.json")
                    break
                except:
                    print("Failed to get steps from LLM")
                    continue
            step_by_step_prompts = steps["steps"]
        else:
            step_by_step_prompts = all_prompts.split("\n")

        ### PROMPT 2, 3, 4, 5
        for prompt in step_by_step_prompts:
            prompt = prompt.strip()
            if prompt == "":
                continue
            for response in game.process_user_query(prompt):
                print(response)
            code = game.export_code()
            with open(implementation_path, "w") as f:
                f.write(code)