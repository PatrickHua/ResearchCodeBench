from pydantic import BaseModel, Field
from typing import List
from core_annotation.models.problem import Problem
import yaml
import os
from core.async_chat_clients import AsyncChatClients
from core.data_classes.llm_type import LLMType
import shutil
from core_annotation.utils.run_shell_command import run_shell_command, check_complete_success
import copy
from core_annotation.models.prediction import TestResult
from core_annotation.utils.sync_folders import ignore_git
from typing import Optional

class PSet(BaseModel):
    folder_name: str = Field(default='pset')
    # description: str
    problems: List[Problem]
    
    @classmethod
    def parse_pset(cls, pset_dir: str, pset: Optional['PSet'] = None) -> 'PSet':
        """
        If pset is not None, only add problems that are not already in the pset.
        """
        # get the folder name
        folder_name = os.path.basename(os.path.normpath(pset_dir))
        # breakpoint()
        existing_problems = [problem.folder_name for problem in pset.problems] if pset is not None else []
        problems = []
        
        for problem_dir in os.listdir(pset_dir):
            if os.path.isdir(os.path.join(pset_dir, problem_dir)):
                if problem_dir in existing_problems:
                    problems.append(pset.problems[existing_problems.index(problem_dir)])
                else:
                    problems.append(Problem.parse_problem(pset_dir, problem_dir))
        return cls(folder_name=folder_name, problems=problems)

    async def solve_all(self, llm_types: List[LLMType], n_completions: int, temperature: float, clients: AsyncChatClients, gen_one: Optional[str] = None):
        for problem in self.problems:
            if gen_one is not None and problem.folder_name != gen_one:
                continue

            await problem.generate_solutions(llm_types, n_completions, temperature, clients)

    def test_all(self, pset_src_folder: str, cache_dir: str, test_one: Optional[str] = None):
        # copy the pset to the cache dir
        pset_cache_dir = os.path.join(cache_dir, self.folder_name)
        os.makedirs(pset_cache_dir, exist_ok=True)
        
        for problem in self.problems:
            if test_one is not None and problem.folder_name != test_one:
                continue

            problem_cache_dir = os.path.join(pset_cache_dir, problem.folder_name)
            problem_src_dir = os.path.join(pset_src_folder, problem.folder_name)
            os.makedirs(problem_cache_dir, exist_ok=True)
            assert os.path.exists(problem_src_dir)
            
            for snippet in problem.problem_file.snippets:
                for llm_type, predictions in snippet.predictions.items():
                    for completion in predictions.completions:
                        if completion.test_result is not None:
                            continue
                        problem_file_code_copy = copy.deepcopy(problem.problem_file.code)
                        problem_file_code_copy.lines[snippet.start_line+1:snippet.end_line] = completion.formatted_completion.lines
                        problem_file_str = str(problem_file_code_copy)
                        problem_file_full_path = os.path.join(cache_dir, self.folder_name, problem.folder_name, problem.problem_file.rel_path)

                        shutil.copytree(problem_src_dir, problem_cache_dir, 
                                        dirs_exist_ok=True, 
                                        ignore=ignore_git)
                        # update the problem file
                        with open(problem_file_full_path, "w") as f:
                            f.write(problem_file_str)

                        # run the test
                        test_header = f"""
import sys
sys.path.append("{os.path.join(cache_dir, self.folder_name, problem.folder_name)}")
"""
                        with open(os.path.join(problem_cache_dir, problem.test_entry_point), "w") as f:
                            # prepend the test header
                            f.write(test_header)
                            f.write(open(os.path.join(problem_src_dir, problem.test_entry_point)).read())

                        run_test_command = f"python {os.path.join(problem_cache_dir, problem.test_entry_point)}"
                        print(run_test_command)
                        # breakpoint()
                        success, exit_code, stdout, stderr = run_shell_command(run_test_command)
                        print(success, exit_code, stdout, stderr)
                        print()
                        passed = check_complete_success(success, exit_code, stdout, stderr)
                        completion.test_result = TestResult(success=success, exit_code=exit_code, stdout=stdout, stderr=stderr, passed=passed)

    def summarize_results(self):
        """
        Summarize the results of the test.
        Print the results of different llm types for each problem and snippet.
        Print the averaged success rate of each llm type - averaged over all problems and snippets.
        success rate of one problem = number of snippets that passed / total number of snippets
        """
        # Dictionary to track success counts for each LLM type
        llm_stats = {}
        
        # Print detailed results and gather statistics
        print("\n=== DETAILED RESULTS ===")
        for problem_idx, problem in enumerate(self.problems, 1):
            print(f"\nProblem {problem_idx}: {problem.folder_name}")
            
            for snippet_idx, snippet in enumerate(problem.problem_file.snippets, 1):
                print(f"  Snippet {snippet_idx}:")
                
                for llm_type, predictions in snippet.predictions.items():
                    # Initialize stats for this LLM type if not already done
                    if any(completion.test_result is None for completion in predictions.completions):
                        continue
                    
                    if llm_type not in llm_stats:
                        llm_stats[llm_type] = {"success": 0, "total": 0}
                    
                    # Calculate success for this snippet
                    successes = sum(1 for completion in predictions.completions if completion.test_result.passed)
                    total = len(predictions.completions)
                    success_rate = (successes / total) * 100 if total > 0 else 0
                    
                    # Update overall stats
                    llm_stats[llm_type]["success"] += successes
                    llm_stats[llm_type]["total"] += total
                    
                    # Print snippet results
                    print(f"    {llm_type}: {successes}/{total} passed ({success_rate:.1f}%)")
                    
                    # Optionally print individual completion results for debugging
                    for comp_idx, completion in enumerate(predictions.completions, 1):
                        result = "✓" if completion.test_result.passed else "✗"
                        exit_code = completion.test_result.exit_code
                        print(f"      {result} Completion {comp_idx} (Exit Code: {exit_code})")
                        # breakpoint()
        # Calculate and print overall success rates
        print("\n=== OVERALL SUCCESS RATES ===")
        for llm_type, stats in sorted(llm_stats.items()):
            success_rate = (stats["success"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            print(f"{llm_type}: {stats['success']}/{stats['total']} passed ({success_rate:.1f}%)")
        
        # Find the best performing LLM
        if llm_stats:
            best_llm = max(llm_stats.items(), 
                          key=lambda x: (x[1]["success"] / x[1]["total"]) if x[1]["total"] > 0 else 0)
            best_rate = (best_llm[1]["success"] / best_llm[1]["total"]) * 100 if best_llm[1]["total"] > 0 else 0
            print(f"\nBest performing LLM: {best_llm[0]} ({best_rate:.1f}%)")
