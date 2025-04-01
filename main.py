#!/usr/bin/env python3

import argparse
import os
import copy
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
    parser.add_argument("--src_folder", type=str, default="./pset/")
    parser.add_argument("--output_file", type=str, default="paper2code_answers.json")
    # parser.add_argument("--output_file", type=str, default="tmp.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--llm_types", nargs="+", type=str, default=[ "gemini-1.5-flash-8b", "gpt-4o-mini", "o3-mini", "gpt-4o"])
    parser.add_argument("--n_completions", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_iter", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--gen", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--run_one", default=None, type=str, help="Only generate solutions for this problem")
    parser.add_argument("--wo_paper", action="store_true")
    # parser.add_argument("--test_all", action="store_true")
    # parser.add_argument("--test_one", default=None, type=str, help="Only test this problem")
    
    parser.add_argument("--summarize_results", action="store_true")
    parser.add_argument("--cache_dir", type=str, default="./.cache/")
    args = parser.parse_args()

    llm_types = [LLMType(name) for name in args.llm_types]
    clients = AsyncChatClients()
    output_file = os.path.join(args.src_folder, args.output_file)
    pset = None
    if os.path.exists(output_file) and not args.overwrite:
        with open(output_file, "r", encoding="utf-8") as f:
            pset = PSet.model_validate_json(f.read())
            if args.run_one is not None:
                pset.problems = [problem for problem in pset.problems if problem.folder_name == args.run_one]
    else:
        pset = PSet.parse_pset(args.src_folder, pset, args.run_one)
    return pset, llm_types, clients, output_file, args

async def main(pset, llm_types, clients, args):


    await pset.solve_all(llm_types, args.n_completions, args.temperature, clients)





if __name__ == '__main__':
    pset, llm_types, clients, output_file, args = parse_args()
    
    if args.gen:

        asyncio.run(main(pset, llm_types, clients, args))

        with open(output_file, "w") as f:
            f.write(pset.model_dump_json(indent=4))
        print(f"Done. Results saved to {output_file}")


    if args.test:
        # copy the pset to the cache dir
        # cache_dir = os.path.join(args.cache_dir, os.path.basename(args.src_folder))
        # shutil.copytree(args.src_folder, cache_dir)  # .cache/pset/
        # breakpoint()
        pset.test_all(args.src_folder, args.cache_dir)

        with open(output_file, "w") as f:
            f.write(pset.model_dump_json(indent=4))
        print(f"Done. Results saved to {output_file}")
    
    
    if args.summarize_results:
        pset.summarize_results()


    print(f"Total inference cost after creating problems: {clients.total_inference_cost}")
    