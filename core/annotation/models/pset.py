from pydantic import BaseModel, Field
from typing import List
from core.annotation.models.problem import Problem
import yaml
import os
from core.async_chat_clients import AsyncChatClients
from core.data_classes.llm_type import LLMType
import shutil
from core.annotation.utils.run_shell_command import run_shell_command, check_complete_success, run_shell_commands_parallel
import copy
from core.annotation.models.prediction import TestResult
from core.annotation.utils.sync_folders import ignore_git
from typing import Optional
import asyncio
import matplotlib.pyplot as plt
import numpy as np
import math
import uuid
from typing import Dict
from tqdm import tqdm
from core.annotation.utils.knowledge_cut_off_date import get_nearest_knowledge_cutoff
import datetime
from datetime import datetime as dt
from core.data_classes.llm_type import LLMType, MODEL_CONFIGS

class PSet(BaseModel):
    folder_name: str = Field(default='pset')
    # description: str
    problems: List[Problem]
    
    @classmethod
    def parse_pset(cls, pset_dir: str, pset: Optional['PSet'] = None, selected_problems: Optional[List[str]] = None) -> 'PSet':
        """
        Parse problems from a directory.
        If pset is provided, update existing problems by adding any newly found files or snippets from the directory.
        Problems found in the directory but not in the input pset are added fully.
        """
        # get the folder name
        folder_name = os.path.basename(os.path.normpath(pset_dir))
        # Use a map for efficient lookup and store copies to avoid modifying the input pset
        existing_problems_map = {prob.folder_name: copy.deepcopy(prob) for prob in pset.problems} if pset is not None else {}
        output_problems = [] # Store the problems for the new PSet object

        paper_yaml_path = os.path.join(pset_dir, "papers.yaml")
        if not os.path.exists(paper_yaml_path):
            raise FileNotFoundError(f"papers.yaml not found in {pset_dir}")
            
        with open(paper_yaml_path, "r") as f:
            papers = yaml.safe_load(f)
        
        found_problem_ids = set()
        
        for paper in papers:
            sub_folder_name = paper["id"]
            if selected_problems is not None and sub_folder_name not in selected_problems:
                continue
        
            problem_dir = os.path.join(pset_dir, sub_folder_name)
            if os.path.isdir(problem_dir):
                found_problem_ids.add(sub_folder_name)
                # Parse the problem details from the directory
                parsed_problem = Problem.parse_problem(pset_dir, sub_folder_name, paper)

                if sub_folder_name in existing_problems_map:
                    # Problem exists in input pset, perform additive merge
                    existing_problem_copy = existing_problems_map[sub_folder_name]
                    
                    # Additive Merge Logic:
                    existing_files_map = {file.rel_path: file for file in existing_problem_copy.problem_files}
                    for parsed_file in parsed_problem.problem_files:
                        if parsed_file.rel_path in existing_files_map:
                            # File exists, check for new snippets within this file
                            existing_file = existing_files_map[parsed_file.rel_path]
                            # Assuming snippet names are unique identifiers within a file
                            existing_snippets_map = {snippet.name: snippet for snippet in existing_file.snippets}
                            for parsed_snippet in parsed_file.snippets:
                                if parsed_snippet.name not in existing_snippets_map:
                                    # Add new snippet found on disk to the existing file
                                    if pset is not None: # Print only if merging
                                        print(f"  + Adding new snippet '{parsed_snippet.name}' to file '{existing_file.rel_path}' in problem '{sub_folder_name}'")
                                    existing_file.snippets.append(parsed_snippet)
                        else:
                            # File is new for this problem, add it
                            if pset is not None: # Print only if merging
                                print(f"  + Adding new file '{parsed_file.rel_path}' to problem '{sub_folder_name}'")
                            existing_problem_copy.problem_files.append(parsed_file)
                            
                    # Add other potential updates to problem metadata if needed (optional)
                    # e.g., existing_problem_copy.description = parsed_problem.description 
                    
                    output_problems.append(existing_problem_copy) # Add updated copy
                else:
                    # Problem is entirely new, add the parsed version
                    output_problems.append(parsed_problem)

        # Ensure we actually found and processed problems based on papers.yaml entries
        if not output_problems:
             # Check if it was due to filtering or genuinely no matching directories
             if selected_problems and not any(pid in found_problem_ids for pid in selected_problems):
                 print(f"Warning: No problems found matching the selection: {selected_problems}")
             elif not found_problem_ids:
                 raise ValueError(f"No valid problem directories found in {pset_dir} corresponding to entries in papers.yaml")
             else:
                 # This case might occur if selected_problems is empty or filtering removed all
                 print(f"Warning: No problems included after processing {pset_dir}. Check selection criteria if used.")


        # If an input pset was given, add back any problems from it that were *not* found in papers.yaml
        # This preserves problems that might exist in the pset object but not (or no longer) on disk
        if pset is not None:
            for existing_prob_name, existing_prob_copy in existing_problems_map.items():
                if existing_prob_name not in found_problem_ids:
                    # Check if it was explicitly excluded by selection
                    if selected_problems is None or existing_prob_name in selected_problems:
                         output_problems.append(existing_prob_copy) # Add the untouched copy back


        # Final check to ensure we have problems if the directory wasn't empty
        if not output_problems and os.listdir(pset_dir):
             # This condition might be too broad, refine if needed.
             # Consider if papers.yaml was empty or only contained non-directory entries.
             print(f"Warning: Resulting problem set is empty despite non-empty directory {pset_dir}. Ensure papers.yaml is correct and directories exist.")
             # We might still return an empty PSet if that's the valid outcome (e.g., all filtered out)
             # The original code had assert len(problems) > 0, let's keep a similar check but allow empty if filtered
             if not selected_problems and found_problem_ids:
                 # If no selection was made, and we found potential problems, but output is empty -> likely an issue
                 raise ValueError(f"Problem parsing resulted in an empty set unexpectedly for {pset_dir}")


        # breakpoint()
        return cls(folder_name=folder_name, problems=output_problems)

    async def solve_all(self, llm_types: List[LLMType], n_completions: int, temperature: float, clients: AsyncChatClients, wo_paper: bool = False):
        # Run all problems concurrently instead of sequentially
        tasks = [
            problem.generate_solutions(llm_types, n_completions, temperature, clients, wo_paper=wo_paper)
            for problem in self.problems
        ]
        await asyncio.gather(*tasks)
    
    async def solve_sequentially(self, llm_types: List[LLMType], n_completions: int, temperature: float, clients: AsyncChatClients, wo_paper: bool = False, output_file: str = None, overwrite_by_prob: Optional[str] = None, overwrite_by_llm: Optional[str] = None):
        # check if overwrite_by_prob is a problem in the pset
        if overwrite_by_prob is not None:
            assert overwrite_by_prob in [problem.folder_name for problem in self.problems], f"Problem {overwrite_by_prob} not found in pset"
        if overwrite_by_llm is not None:
            breakpoint()
            assert overwrite_by_llm in [llm_type.name for llm_type in llm_types], f"LLM type {overwrite_by_llm} not found in LLMType"
            
        for problem in tqdm(self.problems, desc="Solving problems"):
            if overwrite_by_prob is not None and overwrite_by_prob == problem.folder_name:
                overwrite_problem = True
            else:
                overwrite_problem = False
            await problem.generate_solutions(llm_types, n_completions, temperature, clients, wo_paper=wo_paper, overwrite=overwrite_problem, overwrite_by_llm=overwrite_by_llm)
            if output_file is not None:
                with open(output_file, "w") as f:
                    f.write(self.model_dump_json(indent=4))
            

    def test_all(self, pset_src_folder: str, cache_dir: str, overwrite_by_problem: bool = False, parallel: bool = False, max_workers: Optional[int] = None, timeout_seconds: int = 10, output_file: str = None, overwrite_by_llm: Optional[str] = None):
        """
        Test all problems in the problem set.
        
        Args:
            pset_src_folder: Path to the source folder of the problem set
            cache_dir: Path to the cache directory
            overwrite_by_problem: Whether to overwrite existing test results by problem
            parallel: Whether to run tests in parallel
            max_workers: Maximum number of worker threads for parallel execution
            timeout_seconds: Timeout in seconds for each test
        """
        # copy the pset to the cache dir
        pset_cache_dir = os.path.join(cache_dir, self.folder_name)
        os.makedirs(pset_cache_dir, exist_ok=True)

        if parallel:
            raise NotImplementedError("Parallel testing is not implemented")
            print(f"Running tests in parallel mode with timeout {timeout_seconds}s per test...")
            self._test_all_parallel(pset_src_folder, cache_dir, overwrite, max_workers, timeout_seconds)
        else:
            print(f"Running tests sequentially with timeout {timeout_seconds}s per test...")
            self._test_all_sequential(pset_src_folder, cache_dir, overwrite_by_problem, timeout_seconds, output_file, overwrite_by_llm)


    def _test_all_sequential(self, pset_src_folder: str, cache_dir: str, overwrite_by_problem: Optional[str] = None, timeout_seconds: int = 10, output_file: str = None, overwrite_by_llm: Optional[str] = None):
        """Sequential implementation of test_all"""
        pset_cache_dir = os.path.join(cache_dir, self.folder_name)
        
        
        if overwrite_by_problem:
            assert overwrite_by_problem in [problem.folder_name for problem in self.problems], f"Problem {overwrite_by_problem} not found in pset"
        # if overwrite_by_llm:
        #     assert overwrite_by_llm in [llm_type.name for llm_type in LLMType], f"LLM type {overwrite_by_llm} not found in LLMType"
        # breakpoint()
        
        for problem in self.problems:
            problem_cache_dir = os.path.join(pset_cache_dir, problem.folder_name)
            problem_src_dir = os.path.join(pset_src_folder, problem.folder_name)
            os.makedirs(problem_cache_dir, exist_ok=True)
            assert os.path.exists(problem_src_dir)
            
            for problem_file in problem.problem_files:
                for snippet in problem_file.snippets:
                    for llm_type, predictions in snippet.predictions.items():

                        for completion in predictions.completions:
                            if completion.test_result is not None: # if the completion has already been tested, skip it by default, unless we are overwriting by problem or llm
                                # breakpoint()
                                if overwrite_by_problem is None and overwrite_by_llm is not None:
                                    if llm_type == overwrite_by_llm:
                                        pass
                                    else:
                                        continue
                                elif overwrite_by_problem is not None and overwrite_by_llm is None:
                                    if problem.folder_name == overwrite_by_problem:
                                        pass
                                    else:
                                        continue
                                elif overwrite_by_problem is not None and overwrite_by_llm is not None:
                                    if problem.folder_name == overwrite_by_problem and llm_type == overwrite_by_llm:
                                        # breakpoint()
                                        pass
                                    else:
                                        continue
                                else:
                                    continue
                                

                            problem_file_code_copy = copy.deepcopy(problem_file.code)
                            # if snippet.name == "x=?":
                            #     print(problem_file_code_copy.lines)
                            #     breakpoint()
                            print(problem_file_code_copy.lines[snippet.start_line+1:snippet.end_line], completion.formatted_completion.lines)
                            # breakpoint()
                            problem_file_code_copy.lines[snippet.start_line+1:snippet.end_line] = completion.formatted_completion.lines
                            problem_file_str = str(problem_file_code_copy)
                            problem_file_full_path = os.path.join(cache_dir, self.folder_name, problem.folder_name, problem_file.rel_path)

                            shutil.copytree(problem_src_dir, problem_cache_dir, 
                                            dirs_exist_ok=True, 
                                            ignore=ignore_git)
                            # update the problem file
                            with open(problem_file_full_path, "w") as f:
                                f.write(problem_file_str)

                            with open(os.path.join(problem_cache_dir, problem.test_entry_point), "w") as f:
                                f.write(open(os.path.join(problem_src_dir, problem.test_entry_point)).read())

                            run_test_command = f"cd {problem_cache_dir} && timeout {timeout_seconds} python {os.path.join(problem.test_entry_point)}"
                            print(snippet.name, run_test_command)
                            success, exit_code, stdout, stderr = run_shell_command(run_test_command)
                            print(success, exit_code, stdout, stderr)
                            print()
                            passed = check_complete_success(success, exit_code, stdout, stderr)
                            completion.test_result = TestResult(success=success, exit_code=exit_code, stdout=stdout, stderr=stderr, passed=passed)

                            if output_file is not None:
                                with open(output_file, "w") as f:
                                    f.write(self.model_dump_json(indent=4))

    def _test_all_parallel(self, pset_src_folder: str, cache_dir: str, overwrite: bool = False, max_workers: Optional[int] = None, timeout_seconds: int = 10):
        """Parallel implementation of test_all using run_shell_commands_parallel"""
        pset_cache_dir = os.path.join(cache_dir, self.folder_name)
        os.makedirs(pset_cache_dir, exist_ok=True)
        
        # Collect all test tasks
        test_tasks = []
        task_metadata = []
        
        for problem in self.problems:
            problem_src_dir = os.path.join(pset_src_folder, problem.folder_name)
            assert os.path.exists(problem_src_dir), f"Problem source directory not found: {problem_src_dir}"
            
            for problem_file in problem.problem_files:
                for snippet in problem_file.snippets:
                    for llm_type, predictions in snippet.predictions.items():
                        for completion_idx, completion in enumerate(predictions.completions):
                            # skip if the test result is done and overwrite is not set to this problem
                            if completion.test_result is not None and overwrite != problem.folder_name:
                                continue
                            # Create a unique directory for this test to avoid conflicts in parallel execution
                            unique_id = str(uuid.uuid4())[:8]
                            unique_problem_dir = os.path.join(pset_cache_dir, f"{problem.folder_name}_{unique_id}")
                            
                            # Prepare the test
                            test_tasks.append({
                                'problem': problem,
                                'problem_file': problem_file,
                                'snippet': snippet,
                                'llm_type': llm_type,
                                'completion': completion,
                                'completion_idx': completion_idx,
                                'problem_src_dir': problem_src_dir,
                                'unique_problem_dir': unique_problem_dir,
                                'timeout_seconds': timeout_seconds
                            })
        
        if not test_tasks:
            print("No tests to run.")
            return
            
        print(f"Preparing to run {len(test_tasks)} tests in parallel...")
        
        # First create all the unique test directories with the right code
        commands = []
        
        for task in test_tasks:
            # Create unique directory
            os.makedirs(task['unique_problem_dir'], exist_ok=True)
            
            # Copy the problem source files
            shutil.copytree(task['problem_src_dir'], task['unique_problem_dir'], 
                          dirs_exist_ok=True, 
                          ignore=ignore_git)
            
            # Prepare the modified file with the completion
            problem_file_code_copy = copy.deepcopy(task['problem_file'].code)
            snippet = task['snippet']
            completion = task['completion']
            problem_file_code_copy.lines[snippet.start_line+1:snippet.end_line] = completion.formatted_completion.lines
            problem_file_str = str(problem_file_code_copy)
            
            # Write the modified file
            problem_file_path = os.path.join(task['unique_problem_dir'], task['problem_file'].rel_path)
            os.makedirs(os.path.dirname(problem_file_path), exist_ok=True)
            with open(problem_file_path, "w") as f:
                f.write(problem_file_str)
            
            # Ensure the test script is also present
            with open(os.path.join(task['unique_problem_dir'], task['problem'].test_entry_point), "w") as f:
                f.write(open(os.path.join(task['problem_src_dir'], task['problem'].test_entry_point)).read())
            
            # Create the command
            cmd = f"cd {task['unique_problem_dir']} &&  python {os.path.join(task['problem'].test_entry_point)}"
            commands.append(cmd)
            task_metadata.append(task)
        
        # Run all tests in parallel
        print(f"Executing {len(commands)} tests in parallel with {max_workers or 'default'} workers...")
        results = run_shell_commands_parallel(commands, max_workers=max_workers)
        
        # Process results
        for i, (success, exit_code, stdout, stderr) in enumerate(results):
            task = task_metadata[i]
            
            print(f"Result for problem: {task['problem'].folder_name}, snippet: {task['snippet'].name}, "
                  f"LLM: {task['llm_type'].name}, completion: {task['completion_idx']}")
            print(f"Command: {commands[i]}")
            print(f"Success: {success}, Exit code: {exit_code}")
            
            passed = check_complete_success(success, exit_code, stdout, stderr)
            task['completion'].test_result = TestResult(
                success=success, 
                exit_code=exit_code, 
                stdout=stdout, 
                stderr=stderr, 
                passed=passed
            )
            
            # Clean up the unique directory after test is done
            try:
                shutil.rmtree(task['unique_problem_dir'])
            except Exception as e:
                print(f"Warning: Failed to clean up directory {task['unique_problem_dir']}: {e}")
        
        print(f"Completed {len(results)} parallel test executions.")

    def summarize_results(self, n_completions: int, save_to_json: bool = False, output_dir: str = None, contamination_free: bool = False, llm_types: Optional[List[LLMType]] = None):
        """
        Summarize the results of the test.
        Print the results of different llm types for each problem and snippet.
        Print the averaged success rate of each llm type - averaged over all problems and snippets.
        success rate of one problem = number of lines of code that pass the test / total number of lines of code
        
        Args:
            save_to_json: If True, saves detailed results to a JSON file
            output_dir: Path to save output files
        """
        # Dictionary to track success counts for each LLM type
        llm_stats = {}
        # Dictionary to store detailed results for JSON output
        detailed_results = {
            "problems": [],
            "overall_stats": {},
            "best_performing_llm": None
        }
        
        paper_yaml_path = os.path.join(self.folder_name, "papers.yaml")
        with open(paper_yaml_path, "r") as f:
            papers = yaml.load(f, Loader=yaml.BaseLoader)
            # breakpoint()
        papers_dict = {paper["id"]: paper for paper in papers}

        
        # Convert nearest_knowledge_cutoff string (YYYY-MM) to a datetime.date object
        contaminated_problems = []

        if contamination_free:
            for problem in self.problems:
                first_commit_date = papers_dict[problem.folder_name]['first_commit_date']
                nearest_knowledge_cutoff = get_nearest_knowledge_cutoff(llm_types)
                cutoff_year, cutoff_month = map(int, nearest_knowledge_cutoff.split('-'))
                cutoff_date = datetime.date(cutoff_year, cutoff_month, 1)
                
                # Convert first_commit_date string to date object if it's a string
                if isinstance(first_commit_date, str):
                    # Assuming format is something like 'YYYY-MM-DD'
                    year, month, day = map(int, first_commit_date.split('-'))
                    first_commit_date = datetime.date(year, month, day)
                
                # Compare the dates
                is_contaminated = first_commit_date <= cutoff_date
                print(f"First commit date: {first_commit_date}, Knowledge cutoff: {cutoff_date}")
                print(f"Is contaminated: {is_contaminated}")
                if is_contaminated:
                    contaminated_problems.append(problem.folder_name)
            print(f"Found {len(contaminated_problems)} contaminated problems: {contaminated_problems}")
        
        # Print detailed results and gather statistics
        print("\n=== DETAILED RESULTS ===")
        for problem_idx, problem in enumerate(self.problems, 1):
            print(f"\nProblem {problem_idx}: {problem.folder_name}")

            if problem.folder_name in contaminated_problems:
                continue
            
            llm_stats[problem.folder_name] = {"paper_metadata": papers_dict[problem.folder_name], "results": {}}
            
            for problem_file in problem.problem_files:
                for snippet_idx, snippet in enumerate(problem_file.snippets, 1):
                    
                    # Get the number of code lines in this snippet
                    snippet_code_lines = snippet.code.get_code_lines()
                    snippet_code_line_count = len(snippet_code_lines)

                    
                    for llm_type_name, predictions in snippet.predictions.items():
                        # Initialize stats for this LLM type if not already done
                        if any(completion.test_result is None for completion in predictions.completions):
                            continue
                        if llm_type_name not in llm_stats[problem.folder_name]["results"]:
                            llm_stats[problem.folder_name]["results"][llm_type_name] = {"results": {}}
                            # llm_stats[problem.folder_name]["results"][llm_type_name]["llm_cfg"] =
                            # breakpoint()
                        if snippet.name not in llm_stats[problem.folder_name]["results"][llm_type_name]["results"]:
                            llm_stats[problem.folder_name]["results"][llm_type_name]["results"][snippet.name] = []
                        
                        for completion_idx, completion in enumerate(predictions.completions):
                            if completion_idx + 1 > n_completions:
                                break
                            llm_stats[problem.folder_name]["results"][llm_type_name]["results"][snippet.name].append({
                                "snippet_code_line_count": snippet_code_line_count,
                                "completion_idx": completion_idx,
                                "passed": completion.test_result.passed,
                                "exit_code": completion.test_result.exit_code
                            })
                            

        # Dictionary to store success rates for plotting
        problem_llm_success_rates = []
        
        for i in range(n_completions):
            problem_llm_success_rates_per_completion = {}
            for problem_folder_name, problem_stats in llm_stats.items():
                if problem_folder_name in contaminated_problems:
                    continue
                problem_llm_success_rates_per_completion[problem_folder_name] = {}
                
                for llm_type_name, llm_type_stats in problem_stats["results"].items():
                    success_lines: List[int] = []
                    total_lines: List[int] = []
                    
                    for snippet_name, snippet_stats in llm_type_stats["results"].items():
                        # breakpoint()
                        try:

                            if len(snippet_stats) == 0:
                                print(f"#### {i} {problem_folder_name} {llm_type_name} {snippet_name} {snippet_stats} failed")
                            elif not snippet_stats[i]["passed"]:
                                print(f"#### {i} {problem_folder_name} {llm_type_name} {snippet_name} {snippet_stats[i]['snippet_code_line_count']} failed")
                                total_lines.append(snippet_stats[i]["snippet_code_line_count"])
                            else:
                                success_lines.append(snippet_stats[i]["snippet_code_line_count"])
                                print(f"#### {i} {problem_folder_name} {llm_type_name} {snippet_name} {snippet_stats[i]['snippet_code_line_count']} passed")
                            
                                total_lines.append(snippet_stats[i]["snippet_code_line_count"])
                            
                        except Exception as e:
                            print(f"Error processing snippet {snippet_name}: {e}")
                            breakpoint()
                    success_rate_per_problem = 100 * sum(success_lines) / sum(total_lines) if sum(total_lines) > 0 else 0

                    print(f"#### {i} {problem_folder_name} {llm_type_name} {success_lines}/{total_lines} ({success_rate_per_problem:.1f}%)")
                    problem_llm_success_rates_per_completion[problem_folder_name][llm_type_name] = success_rate_per_problem  # Convert to percentage
            problem_llm_success_rates.append(problem_llm_success_rates_per_completion)
            
        # Create visualization
        problem_llm_success_rates_dict = self._create_performance_plots(problem_llm_success_rates, plot_path=os.path.join(output_dir, f"results_summary_{'contamination_free' if contamination_free else ''}.png"))

        # Show overall success rates for each LLM type across all problems
        overall_stats = self._show_overall_success_rates(llm_stats, plot_path=os.path.join(output_dir, f"overall_success_rates_{'contamination_free' if contamination_free else ''}.png"))
        # self._count_snippets_per_llm(llm_stats)
        
        for problem_folder_name, problem_stats in llm_stats.items():
            llm_stats[problem_folder_name]["problem_scores"] = problem_llm_success_rates_dict[problem_folder_name]
            # breakpoint()
        for llm_type_name, llm_type_stats in overall_stats.items():
            overall_stats[llm_type_name]["llm_cfg"] = MODEL_CONFIGS[LLMType[llm_type_name]].model_dump()
            
            # [problem_llm_success_rates[i][problem_folder_name] for i in range(n_completions)]
        # breakpoint()
        llm_stats_json = {'results': llm_stats, 'overall_scores': overall_stats}
        # Count snippets per LLM per completion
        

        # Save results to JSON if requested
        if save_to_json:
            import json
            # Save detailed problem stats
            with open(os.path.join(output_dir, f"results_summary{'_contamination_free' if contamination_free else ''}.json"), "w") as f:
                json.dump(llm_stats, f, indent=2)
            # breakpoint()
            # Save overall stats
            with open(os.path.join(output_dir, f"overall_stats{'_contamination_free' if contamination_free else ''}.json"), "w") as f:
                json.dump(llm_stats_json, f, indent=2)
            
            print(f"\nDetailed results saved to {os.path.join(output_dir, 'results_summary.json')}")
            print(f"Overall statistics saved to {os.path.join(output_dir, 'overall_stats.json')}")
    
# import math
# import numpy as np
# import matplotlib.pyplot as plt
# from typing import List, Dict, Any

    def _create_performance_plots(
        self,
        problem_llm_success_rates: List[Dict[str, Dict[str, float]]],
        plot_path: str
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Plot LLM success rates per problem with error bars.

        Args:
            problem_llm_success_rates:
                A list where each element is a dict mapping
                problem_name -> {llm_type: success_rate, ...}.
            plot_path:
                Filepath to save the resulting PNG.

        Returns:
            {
            problem_name: {
                llm_type_1: {'mean': ..., 'std': ...},
                llm_type_2: {'mean': ..., 'std': ...},
                ...
            },
            ...
            }
        """
        n = len(problem_llm_success_rates)
        if n == 0:
            raise ValueError("No data to visualize.")

        # collect all problems and LLM keys
        all_problems = sorted({p for batch in problem_llm_success_rates for p in batch})
        all_llms     = sorted({
            llm
            for batch in problem_llm_success_rates
            for rates in batch.values()
            for llm in rates
        })
        if not all_problems:
            raise ValueError("No problems with results to visualize.")

        # prepare return dict
        stats: Dict[str, Dict[str, Dict[str, float]]] = {}

        # grid for subplots
        cols = min(4, len(all_problems))
        rows = math.ceil(len(all_problems) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False)

        for idx, problem in enumerate(all_problems):
            row, col = divmod(idx, cols)
            ax = axes[row][col]

            # build per-LLM stats
            per_llm: Dict[str, Dict[str, float]] = {}
            for llm in all_llms:
                vals = [
                    batch[problem][llm]
                    for batch in problem_llm_success_rates
                    if problem in batch and llm in batch[problem]
                ]
                mean = float(np.mean(vals)) if vals else 0.0
                std  = float(np.std(vals))  if vals else 0.0
                per_llm[llm] = {'mean': mean, 'std': std}

            stats[problem] = per_llm

            # plot
            x = range(len(all_llms))
            means = [per_llm[llm]['mean'] for llm in all_llms]
            stds  = [per_llm[llm]['std']  for llm in all_llms]

            ax.bar(x, means, yerr=stds, align='center', capsize=4)
            ax.set_title(problem)
            ax.set_xticks(x)
            ax.set_xticklabels(all_llms, rotation=45, ha='right')
            ax.set_ylim(0, 105)
            ax.set_ylabel("Success Rate (%)")

            # annotate
            for xi, llm in zip(x, all_llms):
                m, s = per_llm[llm]['mean'], per_llm[llm]['std']
                label = f"{m:.1f}%±{s:.1f}" if s > 0 else f"{m:.1f}%"
                ax.text(xi, m + (s if s > 0 else 1), label,
                        ha='center', va='bottom', fontsize=7)

        # hide unused axes
        total = rows * cols
        for extra in range(len(all_problems), total):
            r, c = divmod(extra, cols)
            axes[r][c].axis('off')

        plt.tight_layout()
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        # breakpoint()
        return stats

    def _show_overall_success_rates(self, llm_stats, plot_path: str = "overall_success_rates.png"):
        """
        Display the overall success rates for each LLM type across all problems.
        Success rate is calculated as: (total passed lines) / (total lines) across all problems.
        Includes standard deviation across different completions.
        
        Returns:
            dict: Dictionary containing success rate statistics for each LLM type
        """
        print("\n=== OVERALL SUCCESS RATES ===")
        
        # Track success rates by LLM type and completion
        llm_completion_rates = {}
        
        # First, organize by completion
        for problem_folder_name, problem_stats in llm_stats.items():
            for llm_type_name, llm_type_stats in problem_stats["results"].items():
                if llm_type_name not in llm_completion_rates:
                    llm_completion_rates[llm_type_name] = {}
                
                for snippet_name, snippet_stats in llm_type_stats["results"].items():
                    for completion_stats in snippet_stats:
                        completion_idx = completion_stats["completion_idx"]
                        
                        if completion_idx not in llm_completion_rates[llm_type_name]:
                            llm_completion_rates[llm_type_name][completion_idx] = {
                                "success_lines": 0, 
                                "total_lines": 0,
                                "success_tasks": 0,
                                "total_tasks": 0
                            }
                        
                        line_count = completion_stats["snippet_code_line_count"]
                        llm_completion_rates[llm_type_name][completion_idx]["total_lines"] += line_count
                        llm_completion_rates[llm_type_name][completion_idx]["total_tasks"] += 1
                        if completion_stats["passed"]:
                            llm_completion_rates[llm_type_name][completion_idx]["success_lines"] += line_count
                            llm_completion_rates[llm_type_name][completion_idx]["success_tasks"] += 1
                        
        # Calculate success rates for each completion
        llm_success_rates = {}
        for llm_type_name, completion_stats in llm_completion_rates.items():
            rates = []
            task_rates = []
            for comp_idx, stats in completion_stats.items():
                success_lines = stats["success_lines"]
                total_lines = stats["total_lines"]
                success_tasks = stats["success_tasks"]
                total_tasks = stats["total_tasks"]
                if total_lines > 0:
                    rates.append(success_lines / total_lines * 100)
                if total_tasks > 0:
                    task_rates.append(success_tasks / total_tasks * 100)
            llm_success_rates[llm_type_name] = {}
            if rates:
                llm_success_rates[llm_type_name]["line_rates"] = {
                    "mean": np.mean(rates),
                    "std": np.std(rates),
                    "rates": rates
                }
            if task_rates:
                llm_success_rates[llm_type_name]["task_rates"] = {
                    "mean": np.mean(task_rates),
                    "std": np.std(task_rates),
                    "rates": task_rates
                }
                
        
        # Sort LLMs by mean success rate for better readability
        sorted_llms = sorted(
            llm_success_rates.items(),
            key=lambda x: x[1]["line_rates"]["mean"],
            reverse=True
        )
        
        # Print summary table
        print(f"{'LLM Type':<20} {'Mean Rate':<15} {'Std Dev':<15} {'Min':<10} {'Max':<10}")
        print("-" * 70)
        
        for llm_type_name, stats in sorted_llms:
            mean_rate = stats["line_rates"]["mean"]
            std_dev = stats["line_rates"]["std"]
            min_rate = min(stats["line_rates"]["rates"])
            max_rate = max(stats["line_rates"]["rates"])
            
            print(f"{llm_type_name:<20} {mean_rate:.2f}%±{std_dev:.2f} {'':<5} {min_rate:.2f}% {'':<3} {max_rate:.2f}%")


        # Sort LLMs by mean success rate for better readability
        sorted_llms = sorted(
            llm_success_rates.items(),
            key=lambda x: x[1]["task_rates"]["mean"],
            reverse=True
        )
        
        # Print summary table
        print(f"{'LLM Type':<20} {'Mean Rate':<15} {'Std Dev':<15} {'Min':<10} {'Max':<10}")
        print("-" * 70)
        
        for llm_type_name, stats in sorted_llms:
            mean_rate = stats["task_rates"]["mean"]
            std_dev = stats["task_rates"]["std"]
            min_rate = min(stats["task_rates"]["rates"])
            max_rate = max(stats["task_rates"]["rates"])
            
            print(f"{llm_type_name:<20} {mean_rate:.2f}%±{std_dev:.2f} {'':<5} {min_rate:.2f}% {'':<3} {max_rate:.2f}%")
        
        
        
        # Create bar chart visualization with error bars
        if sorted_llms:
            self._plot_overall_success_rates(sorted_llms, plot_path=plot_path, rate_type="line_rates")
            self._plot_overall_success_rates(sorted_llms, plot_path=plot_path.replace(".png", "_tasks.png"), rate_type="task_rates")
        # breakpoint()
        return llm_success_rates

    def _count_snippets_per_llm(self, llm_stats):
        """
        Count the total number of snippets attempted per LLM per completion.
        This helps understand the distribution of test attempts across different models.
        """
        print("\n=== SNIPPET COUNT PER LLM ===")
        
        # Track snippets per LLM and completion
        snippet_counts = {}
        
        # Gather statistics
        for problem_folder_name, problem_stats in llm_stats.items():
            for llm_type_name, llm_type_stats in problem_stats.items():
                if llm_type_name not in snippet_counts:
                    snippet_counts[llm_type_name] = {}
                
                for snippet_name, snippet_stats in llm_type_stats.items():
                    for completion_stats in snippet_stats:
                        completion_idx = completion_stats["completion_idx"]
                        if completion_idx not in snippet_counts[llm_type_name]:
                            snippet_counts[llm_type_name][completion_idx] = 0
                        snippet_counts[llm_type_name][completion_idx] += 1
        
        # Print summary
        for llm_type_name, completion_counts in sorted(snippet_counts.items()):
            print(f"\nLLM: {llm_type_name}")
            print(f"{'Completion #':<15} {'Snippet Count':<15}")
            print("-" * 30)
            
            # Calculate total for this LLM
            total_snippets = sum(completion_counts.values())
            
            # Print per-completion breakdown
            for completion_idx, count in sorted(completion_counts.items()):
                print(f"{completion_idx:<15} {count:<15}")
            
            # Print total
            print(f"{'TOTAL':<15} {total_snippets:<15}")
            
        # Print grand total across all LLMs
        all_llm_total = sum(sum(completions.values()) for completions in snippet_counts.values())
        print(f"\nTotal snippets across all LLMs: {all_llm_total}")

    def _plot_overall_success_rates(self, sorted_llms, plot_path: str = "overall_success_rates.png", rate_type: str = "line_rates"):
        """
        Create a bar chart visualization of overall success rates with error bars.
        
        Args:
            sorted_llms: List of (llm_name, stats) tuples sorted by success rate
        """
        # Extract data for plotting
        llm_names = [llm[0] for llm in sorted_llms]
        means = [llm[1][rate_type]["mean"] for llm in sorted_llms]
        stds = [llm[1][rate_type]["std"] for llm in sorted_llms]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot bar chart with error bars
        x_pos = np.arange(len(llm_names))
        yerr = [std if std > 0 else np.nan for std in stds]
        bars = plt.bar(x_pos, means, yerr=yerr, align='center', 
                     color='skyblue', alpha=0.7, ecolor='black', capsize=5)
        
        # Add labels and customize
        plt.title("Overall Success Rates by LLM Type")
        plt.xlabel("LLM Type")
        plt.ylabel("Success Rate (%)")
        plt.xticks(x_pos, llm_names, rotation=45, ha='right')
        plt.ylim(0, 105)  # 0-100% with a little margin
        
        # Add value labels on top of bars
        for bar, mean, std in zip(bars, means, stds):
            if std > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                        f"{mean:.1f}%±{std:.1f}", ha='center', va='bottom', fontsize=9)
            else:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f"{mean:.1f}%", ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Overall success rates visualization saved to {plot_path}")
        plt.close()
