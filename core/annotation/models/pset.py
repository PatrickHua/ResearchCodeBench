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

class PSet(BaseModel):
    folder_name: str = Field(default='pset')
    # description: str
    problems: List[Problem]
    
    @classmethod
    def parse_pset(cls, pset_dir: str, pset: Optional['PSet'] = None, selected_problems: Optional[List[str]] = None) -> 'PSet':
        """
        If pset is not None, only add problems that are not already in the pset.
        """
        # get the folder name
        folder_name = os.path.basename(os.path.normpath(pset_dir))
        existing_problems = [problem.folder_name for problem in pset.problems] if pset is not None else []
        problems = []
        

        
        paper_yaml_path = os.path.join(pset_dir, "papers.yaml")
        with open(paper_yaml_path, "r") as f:
            papers = yaml.safe_load(f)
        
        for paper in papers:
            folder_name = paper["id"]
            if selected_problems is not None and folder_name not in selected_problems:
                continue
        
            problem_dir = os.path.join(pset_dir, folder_name)
            if os.path.isdir(problem_dir):
                if folder_name in existing_problems:
                    problems.append(pset.problems[existing_problems.index(folder_name)])
                else:
                    problems.append(Problem.parse_problem(pset_dir, folder_name, paper))

        assert len(problems) > 0, f"No problems found in {pset_dir}"
        
        return cls(folder_name=folder_name, problems=problems)

    async def solve_all(self, llm_types: List[LLMType], n_completions: int, temperature: float, clients: AsyncChatClients, wo_paper: bool = False):
        # Run all problems concurrently instead of sequentially
        tasks = [
            problem.generate_solutions(llm_types, n_completions, temperature, clients, wo_paper=wo_paper)
            for problem in self.problems
        ]
        await asyncio.gather(*tasks)
    
    async def solve_sequentially(self, llm_types: List[LLMType], n_completions: int, temperature: float, clients: AsyncChatClients, wo_paper: bool = False, output_file: str = None):
        for problem in tqdm(self.problems, desc="Solving problems"):
            await problem.generate_solutions(llm_types, n_completions, temperature, clients, wo_paper=wo_paper)
            if output_file is not None:
                with open(output_file, "w") as f:
                    f.write(self.model_dump_json(indent=4))
            

    def test_all(self, pset_src_folder: str, cache_dir: str, overwrite: bool = False, parallel: bool = False, max_workers: Optional[int] = None, timeout_seconds: int = 10, output_file: str = None, overwrite_by_llm: Optional[str] = None):
        """
        Test all problems in the problem set.
        
        Args:
            pset_src_folder: Path to the source folder of the problem set
            cache_dir: Path to the cache directory
            overwrite: Whether to overwrite existing test results
            parallel: Whether to run tests in parallel
            max_workers: Maximum number of worker threads for parallel execution
            timeout_seconds: Timeout in seconds for each test
        """
        # copy the pset to the cache dir
        pset_cache_dir = os.path.join(cache_dir, self.folder_name)
        os.makedirs(pset_cache_dir, exist_ok=True)

        if parallel:
            print(f"Running tests in parallel mode with timeout {timeout_seconds}s per test...")
            self._test_all_parallel(pset_src_folder, cache_dir, overwrite, max_workers, timeout_seconds)
        else:
            print(f"Running tests sequentially with timeout {timeout_seconds}s per test...")
            self._test_all_sequential(pset_src_folder, cache_dir, overwrite, timeout_seconds, output_file, overwrite_by_llm)


    def _test_all_sequential(self, pset_src_folder: str, cache_dir: str, overwrite: bool = False, timeout_seconds: int = 10, output_file: str = None, overwrite_by_llm: Optional[str] = None):
        """Sequential implementation of test_all"""
        pset_cache_dir = os.path.join(cache_dir, self.folder_name)
        
        for problem in self.problems:
            problem_cache_dir = os.path.join(pset_cache_dir, problem.folder_name)
            problem_src_dir = os.path.join(pset_src_folder, problem.folder_name)
            os.makedirs(problem_cache_dir, exist_ok=True)
            assert os.path.exists(problem_src_dir)
            
            for problem_file in problem.problem_files:
                for snippet in problem_file.snippets:
                    for llm_type, predictions in snippet.predictions.items():
                        # breakpoint()
                        if overwrite_by_llm is not None and llm_type != overwrite_by_llm:
                            continue
                        for completion in predictions.completions:
                            if completion.test_result is not None and overwrite != problem.folder_name:
                                continue
                            problem_file_code_copy = copy.deepcopy(problem_file.code)
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
                            print(run_test_command)
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

    def summarize_results(self, n_completions: int, save_to_json: bool = False, json_path: str = "results_summary.json"):
        """
        Summarize the results of the test.
        Print the results of different llm types for each problem and snippet.
        Print the averaged success rate of each llm type - averaged over all problems and snippets.
        success rate of one problem = number of lines of code that pass the test / total number of lines of code
        
        Args:
            save_to_json: If True, saves detailed results to a JSON file
            json_path: Path to save the JSON file if save_to_json is True
        """
        # Dictionary to track success counts for each LLM type
        llm_stats = {}
        
        # Dictionary to store detailed results for JSON output
        detailed_results = {
            "problems": [],
            "overall_stats": {},
            "best_performing_llm": None
        }
        
        # Print detailed results and gather statistics
        print("\n=== DETAILED RESULTS ===")
        for problem_idx, problem in enumerate(self.problems, 1):
            print(f"\nProblem {problem_idx}: {problem.folder_name}")
            
            llm_stats[problem.folder_name] = {}
            
            for problem_file in problem.problem_files:
                for snippet_idx, snippet in enumerate(problem_file.snippets, 1):
                    # print(f"\n  Snippet {snippet_idx} ({snippet.name}):")
                    # llm_stats[problem.folder_name][snippet.name] = {}
                    # Get the number of code lines in this snippet
                    snippet_code_lines = snippet.code.get_code_lines()
                    snippet_code_line_count = len(snippet_code_lines)

                    
                    for llm_type_name, predictions in snippet.predictions.items():
                        # Initialize stats for this LLM type if not already done
                        if any(completion.test_result is None for completion in predictions.completions):
                            continue
                        if llm_type_name not in llm_stats[problem.folder_name]:
                            llm_stats[problem.folder_name][llm_type_name] = {}
                        if snippet.name not in llm_stats[problem.folder_name][llm_type_name]:
                            llm_stats[problem.folder_name][llm_type_name][snippet.name] = []
                        
                        for completion_idx, completion in enumerate(predictions.completions):
                            if completion_idx + 1 > n_completions:
                                break
                            llm_stats[problem.folder_name][llm_type_name][snippet.name].append({
                                "snippet_code_line_count": snippet_code_line_count,
                                "completion_idx": completion_idx,
                                "passed": completion.test_result.passed,
                                "exit_code": completion.test_result.exit_code
                            })
                            

        # Dictionary to store success rates for plotting
        problem_llm_success_rates = []
        
        # benchmark_results = {}
        
        
        for i in range(n_completions):
            problem_llm_success_rates_per_completion = {}
            for problem_folder_name, problem_stats in llm_stats.items():
                problem_llm_success_rates_per_completion[problem_folder_name] = {}
                
                for llm_type_name, llm_type_stats in problem_stats.items():
                    success_lines: List[int] = []
                    total_lines: List[int] = []
                    
                    for snippet_name, snippet_stats in llm_type_stats.items():
                        
                        if snippet_stats[i]["passed"]:
                            success_lines.append(snippet_stats[i]["snippet_code_line_count"])
                            print(f"#### {i} {problem_folder_name} {llm_type_name} {snippet_name} {snippet_stats[i]['snippet_code_line_count']} passed")
                        else:
                            print(f"#### {i} {problem_folder_name} {llm_type_name} {snippet_name} {snippet_stats[i]['snippet_code_line_count']} failed")
                        total_lines.append(snippet_stats[i]["snippet_code_line_count"])
                        
                    success_rate_per_problem = 100 * sum(success_lines) / sum(total_lines) if sum(total_lines) > 0 else 0

                    print(f"#### {i} {problem_folder_name} {llm_type_name} {success_lines}/{total_lines} ({success_rate_per_problem:.1f}%)")
                    problem_llm_success_rates_per_completion[problem_folder_name][llm_type_name] = success_rate_per_problem  # Convert to percentage
            problem_llm_success_rates.append(problem_llm_success_rates_per_completion)
        # for i in range(n_completions):
        #     print(problem_llm_success_rates[i]["OptimalSteps"]['O3_MINI_HIGH'])
        # breakpoint()
        # Create visualization
        self._create_performance_plots(problem_llm_success_rates, json_path)

        # Show overall success rates for each LLM type across all problems
        self._show_overall_success_rates(llm_stats)
        
        # Count snippets per LLM per completion
        self._count_snippets_per_llm(llm_stats)

        # Save results to JSON if requested
        if save_to_json:
            import json
            with open(json_path, "w") as f:
                json.dump(llm_stats, f, indent=2)
            print(f"\nDetailed results saved to {json_path}")
    
    def _create_performance_plots(self, problem_llm_success_rates, json_path):
        """
        Create a visualization of LLM performance for each problem with error bars.
        
        Args:
            problem_llm_success_rates: List of dictionaries mapping problem names to LLM success rates
            json_path: Path to use for saving the plot (will replace .json with .png)
        """
        num_completions = len(problem_llm_success_rates)
        
        if num_completions == 0:
            print("No data to visualize.")
            return
            
        # Get all unique problem names and LLM types across all completions
        all_problems = set()
        all_llm_types = set()
        
        for completion_results in problem_llm_success_rates:
            all_problems.update(completion_results.keys())
            for problem, llm_rates in completion_results.items():
                all_llm_types.update(llm_rates.keys())
        
        all_problems = sorted(list(all_problems))
        all_llm_types = sorted(list(all_llm_types))
        
        num_problems = len(all_problems)
        if num_problems == 0:
            print("No problems with results to visualize.")
            return
            
        # Calculate grid dimensions for subplots
        cols = min(4, num_problems)  # Max 4 columns
        rows = math.ceil(num_problems / cols)
        
        # Create figure and subplots
        plt.figure(figsize=(cols * 5, rows * 4))
        
        for i, problem_name in enumerate(all_problems, 1):
            # Calculate mean and standard deviation for each LLM type
            means = []
            stds = []
            
            for llm_type in all_llm_types:
                # Collect rates for this problem and LLM type across all completions
                rates = []
                for completion_results in problem_llm_success_rates:
                    if problem_name in completion_results and llm_type in completion_results[problem_name]:
                        rates.append(completion_results[problem_name][llm_type])
                
                if rates:
                    means.append(np.mean(rates))
                    stds.append(np.std(rates))
                else:
                    means.append(0)
                    stds.append(0)
            
            # Create subplot
            ax = plt.subplot(rows, cols, i)
            
            # Plot bar chart with error bars
            x_pos = np.arange(len(all_llm_types))
            bars = ax.bar(x_pos, means, yerr=stds, align='center', 
                         color='skyblue', alpha=0.7, ecolor='black', capsize=5)
            
            # Add labels and customize
            ax.set_title(f"Problem: {problem_name}")
            ax.set_xlabel("LLM Type")
            ax.set_ylabel("Success Rate (%)")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(all_llm_types, rotation=45, ha='right')
            ax.set_ylim(0, 105)  # 0-100% with a little margin
            
            # Add value labels on top of bars
            for bar, mean, std in zip(bars, means, stds):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                        f"{mean:.1f}%±{std:.1f}", ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = json_path.replace('.json', '.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Performance visualization saved to {plot_path}")
        plt.close()

    def _show_overall_success_rates(self, llm_stats):
        """
        Display the overall success rates for each LLM type across all problems.
        Success rate is calculated as: (total passed lines) / (total lines) across all problems.
        Includes standard deviation across different completions.
        """
        print("\n=== OVERALL SUCCESS RATES ===")
        
        # Track success rates by LLM type and completion
        llm_completion_rates = {}
        
        # First, organize by completion
        for problem_folder_name, problem_stats in llm_stats.items():
            for llm_type_name, llm_type_stats in problem_stats.items():
                if llm_type_name not in llm_completion_rates:
                    llm_completion_rates[llm_type_name] = {}
                
                for snippet_name, snippet_stats in llm_type_stats.items():
                    for completion_stats in snippet_stats:
                        completion_idx = completion_stats["completion_idx"]
                        
                        if completion_idx not in llm_completion_rates[llm_type_name]:
                            llm_completion_rates[llm_type_name][completion_idx] = {
                                "success_lines": 0, 
                                "total_lines": 0
                            }
                        
                        line_count = completion_stats["snippet_code_line_count"]
                        llm_completion_rates[llm_type_name][completion_idx]["total_lines"] += line_count
                        if completion_stats["passed"]:
                            llm_completion_rates[llm_type_name][completion_idx]["success_lines"] += line_count
        
        # Calculate success rates for each completion
        llm_success_rates = {}
        for llm_type_name, completion_stats in llm_completion_rates.items():
            rates = []
            for comp_idx, stats in completion_stats.items():
                success_lines = stats["success_lines"]
                total_lines = stats["total_lines"]
                if total_lines > 0:
                    rates.append(success_lines / total_lines * 100)
            
            if rates:
                llm_success_rates[llm_type_name] = {
                    "mean": np.mean(rates),
                    "std": np.std(rates),
                    "rates": rates
                }
        
        # Sort LLMs by mean success rate for better readability
        sorted_llms = sorted(
            llm_success_rates.items(),
            key=lambda x: x[1]["mean"],
            reverse=True
        )
        
        # Print summary table
        print(f"{'LLM Type':<20} {'Mean Rate':<15} {'Std Dev':<15} {'Min':<10} {'Max':<10}")
        print("-" * 70)
        
        for llm_type_name, stats in sorted_llms:
            mean_rate = stats["mean"]
            std_dev = stats["std"]
            min_rate = min(stats["rates"])
            max_rate = max(stats["rates"])
            
            print(f"{llm_type_name:<20} {mean_rate:.2f}%±{std_dev:.2f} {'':<5} {min_rate:.2f}% {'':<3} {max_rate:.2f}%")
        
        # Create bar chart visualization with error bars
        if sorted_llms:
            self._plot_overall_success_rates(sorted_llms)

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

    def _plot_overall_success_rates(self, sorted_llms):
        """
        Create a bar chart visualization of overall success rates with error bars.
        
        Args:
            sorted_llms: List of (llm_name, stats) tuples sorted by success rate
        """
        # Extract data for plotting
        llm_names = [llm[0] for llm in sorted_llms]
        means = [llm[1]["mean"] for llm in sorted_llms]
        stds = [llm[1]["std"] for llm in sorted_llms]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot bar chart with error bars
        x_pos = np.arange(len(llm_names))
        bars = plt.bar(x_pos, means, yerr=stds, align='center', 
                     color='skyblue', alpha=0.7, ecolor='black', capsize=5)
        
        # Add labels and customize
        plt.title("Overall Success Rates by LLM Type")
        plt.xlabel("LLM Type")
        plt.ylabel("Success Rate (%)")
        plt.xticks(x_pos, llm_names, rotation=45, ha='right')
        plt.ylim(0, 105)  # 0-100% with a little margin
        
        # Add value labels on top of bars
        for bar, mean, std in zip(bars, means, stds):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                    f"{mean:.1f}%±{std:.1f}", ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = "overall_success_rates.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Overall success rates visualization saved to {plot_path}")
        plt.close()
