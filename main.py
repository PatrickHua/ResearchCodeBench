#!/usr/bin/env python3

import argparse
import os
import copy
from core.data_classes.llm_type import LLMType
from core.async_chat_clients import AsyncChatClients
import asyncio

# from core_annotation.models.repo import Repo
from core_annotation.models.pset import PSet
# from paper2code_run import run_shell_command, _save_file_code
import shutil
# from core_annotation.utils.generate_predictions import generate_predictions

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_folder", type=str, default="./pset/")
    # parser.add_argument("--output_file", type=str, default="tmp.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--llm_types", nargs="+", type=str, default=[ "gemini-1.5-flash-8b", "gpt-4o-mini", "o3-mini", "gpt-4o"])
    parser.add_argument("--n_completions", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max_iter", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--gen_all", action="store_true")
    parser.add_argument("--gen_one", default=None, type=str, help="Only generate solutions for this problem")
    parser.add_argument("--test_all", action="store_true")
    parser.add_argument("--test_one", default=None, type=str, help="Only test this problem")
    
    parser.add_argument("--summarize_results", action="store_true")
    parser.add_argument("--cache_dir", type=str, default="./.cache/")
    args = parser.parse_args()

    llm_types = [LLMType(name) for name in args.llm_types]
    clients = AsyncChatClients()
    output_file = os.path.join(args.src_folder, "paper2code_pset_parsed.json")
    pset = None
    if os.path.exists(output_file) and not args.overwrite:
        with open(output_file, "r", encoding="utf-8") as f:
            pset = PSet.model_validate_json(f.read())
    # else:
    pset = PSet.parse_pset(args.src_folder, pset)
    return pset, llm_types, clients, output_file, args

async def main(pset, llm_types, clients, output_file, args):


    await pset.solve_all(llm_types, args.n_completions, args.temperature, clients, args.gen_one)





if __name__ == '__main__':
    pset, llm_types, clients, output_file, args = parse_args()
    
    if args.gen_all or args.gen_one:
        asyncio.run(main(pset, llm_types, clients, output_file, args))

        with open(output_file, "w") as f:
            f.write(pset.model_dump_json(indent=4))
        print(f"Done. Results saved to {output_file}")

    print(f"Total inference cost after creating problems: {clients.total_inference_cost}")
    
    if args.test_all or args.test_one:
        # copy the pset to the cache dir
        # cache_dir = os.path.join(args.cache_dir, os.path.basename(args.src_folder))
        # shutil.copytree(args.src_folder, cache_dir)  # .cache/pset/
        # breakpoint()
        pset.test_all(args.src_folder, args.cache_dir, args.test_one)

        with open(output_file, "w") as f:
            f.write(pset.model_dump_json(indent=4))
        print(f"Done. Results saved to {output_file}")
    
    
    if args.summarize_results:
        pset.summarize_results()


