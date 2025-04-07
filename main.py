#!/usr/bin/env python3

import argparse
import os
import copy
import sys
import time
from core.data_classes.llm_type import LLMType
from core.async_chat_clients import AsyncChatClients
import asyncio

# from core_annotation.models.repo import Repo
from core.annotation.models.pset import PSet
# from paper2code_run import run_shell_command, _save_file_code
import shutil
# from core_annotation.utils.generate_predictions import generate_predictions

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="./pset/")
    parser.add_argument("--output_dir", type=str, default="./outputs/")
    parser.add_argument("--output_file", type=str, default="paper2code_answers.json")
    # parser.add_argument("--output_file", type=str, default="tmp.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--llm_types", nargs="+", type=str, default=[LLMType.GEMINI_1_5_FLASH_8B.value, LLMType.GPT_4O_MINI.value, LLMType.O3_MINI.value, LLMType.GPT_4O.value])
    parser.add_argument("--n_completions", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_iter", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--gen", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--problems", default=None, nargs="+", help="Name of the problems to run. None means all.")
    parser.add_argument("--wo_paper", action="store_true")
    # parser.add_argument("--test_all", action="store_true")
    # parser.add_argument("--test_one", default=None, type=str, help="Only test this problem")
    
    parser.add_argument("--summarize_results", action="store_true")
    parser.add_argument("--cache_dir", type=str, default="./.cache/")
    args = parser.parse_args()

    # Save the command used to run this script
    os.makedirs(args.output_dir, exist_ok=True)
    run_command = " ".join(sys.argv)
    
    with open(os.path.join(args.output_dir, "run.sh"), "w") as f:
        f.write(f"#!/bin/bash\n{sys.executable} {run_command}\n")

    os.chmod(os.path.join(args.output_dir, "run.sh"), 0o755)  # Make it executable

    # Convert string arguments to LLMType enum values
    llm_types = []
    for name in args.llm_types:
        try:
            # First try to get the enum by name
            llm_type = LLMType[name]
        except KeyError:
            try:
                # If that fails, try to get it by value
                llm_type = LLMType(name)
            except ValueError:
                raise ValueError(f"Invalid LLM type: {name}. Valid values are: {[e.name for e in LLMType]}")
        llm_types.append(llm_type)
    
    clients = AsyncChatClients()
    output_file = os.path.join(args.output_dir, args.output_file)
    pset = None
    
    if os.path.exists(output_file) and not args.overwrite:
        # Create backup of output file before reading
        backup_file = f"{output_file}.backup"
        shutil.copy2(output_file, backup_file)
        
        with open(output_file, "r", encoding="utf-8") as f:
            pset = PSet.model_validate_json(f.read())
            if args.problems is not None:
                pset.problems = [problem for problem in pset.problems if problem.folder_name in args.problems]
    # else: 
    pset = PSet.parse_pset(args.data_folder, pset, args.problems)
    return pset, llm_types, clients, output_file, args

async def main(pset, llm_types, clients, args):


    await pset.solve_all(llm_types, args.n_completions, args.temperature, clients)





if __name__ == '__main__':
    pset, llm_types, clients, output_file, args = parse_args()
    
    gen_time = 0
    test_time = 0
    
    if args.gen:
        start_time = time.time()
        asyncio.run(main(pset, llm_types, clients, args))

        # # Create backup of output file
        # backup_file = f"{output_file}.backup"
        # shutil.copy2(output_file, backup_file)
        
        with open(output_file, "w") as f:
            f.write(pset.model_dump_json(indent=4))
        print(f"Done. Results saved to {output_file}")
        gen_time = time.time() - start_time

    if args.test:
        start_time = time.time()
        pset.test_all(args.data_folder, args.cache_dir)

        with open(output_file, "w") as f:
            f.write(pset.model_dump_json(indent=4))
        print(f"Done. Results saved to {output_file}")
        test_time = time.time() - start_time
    
    if args.summarize_results:
        pset.summarize_results(save_to_json=True, json_path=os.path.join(args.output_dir, "results_summary.json"))

    print(f"Total inference cost after creating problems: {clients.total_inference_cost}")
    
    # Print timing information
    if args.gen and args.test:
        print(f"\nExecution times:")
        print(f"Generation time: {gen_time:.2f} seconds")
        print(f"Testing time: {test_time:.2f} seconds")
        print(f"Total time: {gen_time + test_time:.2f} seconds")
    elif args.gen:
        print(f"\nGeneration time: {gen_time:.2f} seconds")
    elif args.test:
        print(f"\nTesting time: {test_time:.2f} seconds")
    