from pydantic import BaseModel, Field
from typing import List
from core.annotation.models.problem import Problem
import yaml
import os
from core.async_chat_clients import AsyncChatClients
from core.data_classes.llm_type import LLMType
import shutil
from core.annotation.utils.run_shell_command import run_shell_command, check_complete_success
import copy
from core.annotation.models.prediction import TestResult
from core.annotation.utils.sync_folders import ignore_git
from typing import Optional
import asyncio

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

    async def solve_all(self, llm_types: List[LLMType], n_completions: int, temperature: float, clients: AsyncChatClients):
        # Run all problems concurrently instead of sequentially
        tasks = [
            problem.generate_solutions(llm_types, n_completions, temperature, clients)
            for problem in self.problems
        ]
        await asyncio.gather(*tasks)

    def test_all(self, pset_src_folder: str, cache_dir: str):
        # copy the pset to the cache dir
        pset_cache_dir = os.path.join(cache_dir, self.folder_name)
        os.makedirs(pset_cache_dir, exist_ok=True)
        
        for problem in self.problems:

            problem_cache_dir = os.path.join(pset_cache_dir, problem.folder_name)
            problem_src_dir = os.path.join(pset_src_folder, problem.folder_name)
            os.makedirs(problem_cache_dir, exist_ok=True)
            assert os.path.exists(problem_src_dir)
            
            for problem_file in problem.problem_files:
                for snippet in problem_file.snippets:
                    for llm_type, predictions in snippet.predictions.items():
                        for completion in predictions.completions:
                            if completion.test_result is not None:
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

                            run_test_command = f"cd {problem_cache_dir} && python {os.path.join(problem.test_entry_point)}"
                            print(run_test_command)
                            success, exit_code, stdout, stderr = run_shell_command(run_test_command)
                            print(success, exit_code, stdout, stderr)
                            print()
                            passed = check_complete_success(success, exit_code, stdout, stderr)
                            completion.test_result = TestResult(success=success, exit_code=exit_code, stdout=stdout, stderr=stderr, passed=passed)

    def summarize_results(self, save_to_json: bool = False, json_path: str = "results_summary.json"):
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
            
            # Calculate total code lines for this problem
            total_code_lines = 0
            for problem_file in problem.problem_files:
                for snippet in problem_file.snippets:
                    # Get the number of code lines (excluding comments and empty lines)
                    code_lines = snippet.code.get_code_lines()
                    total_code_lines += len(code_lines)
            
            # Track code lines that pass for each LLM type
            problem_llm_stats = {}
            
            # Store problem-level results for JSON
            problem_results = {
                "problem_name": problem.folder_name,
                "snippets": [],
                "llm_stats": {}
            }
            
            for problem_file in problem.problem_files:
                for snippet_idx, snippet in enumerate(problem_file.snippets, 1):
                    print(f"\n  Snippet {snippet_idx} ({snippet.name}):")
                    
                    # Get the number of code lines in this snippet
                    snippet_code_lines = snippet.code.get_code_lines()
                    snippet_code_line_count = len(snippet_code_lines)
                    
                    # Store snippet-level results for JSON
                    snippet_results = {
                        "snippet_name": snippet.name,
                        "total_lines": snippet_code_line_count,
                        "llm_results": {}
                    }
                    
                    for llm_type, predictions in snippet.predictions.items():
                        
                        # Initialize stats for this LLM type if not already done
                        if any(completion.test_result is None for completion in predictions.completions):
                            continue
                        
                        if llm_type.name not in llm_stats:
                            llm_stats[llm_type.name] = {"success_lines": 0, "total_lines": 0}
                        
                        if llm_type.name not in problem_llm_stats:
                            problem_llm_stats[llm_type.name] = {"success_lines": 0, "total_lines": 0}
                        
                        # Calculate success for this snippet
                        successes = sum(1 for completion in predictions.completions if completion.test_result.passed)
                        total = len(predictions.completions)
                        success_rate = (successes / total) * 100 if total > 0 else 0

                        # Calculate lines of code that pass
                        success_lines = snippet_code_line_count if successes > 0 else 0
                        
                        # Update overall stats
                        llm_stats[llm_type.name]["success_lines"] += success_lines
                        llm_stats[llm_type.name]["total_lines"] += snippet_code_line_count
                        
                        # Update problem stats
                        problem_llm_stats[llm_type.name]["success_lines"] += success_lines
                        problem_llm_stats[llm_type.name]["total_lines"] += snippet_code_line_count
                        
                        # Store LLM results for JSON
                        llm_results = {
                            "successes": successes,
                            "total": total,
                            "success_rate": success_rate,
                            "success_lines": success_lines,
                            "total_lines": snippet_code_line_count,
                            "completions": []
                        }
                        
                        # Store individual completion results
                        for comp_idx, completion in enumerate(predictions.completions, 1):
                            result = "✓" if completion.test_result.passed else "✗"
                            llm_results["completions"].append({
                                "completion_idx": comp_idx,
                                "passed": completion.test_result.passed,
                                "exit_code": completion.test_result.exit_code
                            })
                            
                            # Print individual completion results for debugging
                            print(f"      {result} Completion {comp_idx} (Exit Code: {completion.test_result.exit_code})")
                        
                        snippet_results["llm_results"][llm_type.name] = llm_results
                        
                        # Print snippet results
                        print(f"    {llm_type.name}: {successes}/{total} passed ({success_rate:.1f}%)")
                        print(f"    {llm_type.name}: {success_lines}/{snippet_code_line_count} lines passed ({success_lines/snippet_code_line_count*100:.1f}% of lines)")
                    
                    problem_results["snippets"].append(snippet_results)
            
            # Store problem-level LLM stats for JSON
            for llm_type, stats in problem_llm_stats.items():
                success_lines = stats["success_lines"]
                total_lines = stats["total_lines"]
                success_rate = (success_lines / total_lines) * 100 if total_lines > 0 else 0
                problem_results["llm_stats"][llm_type] = {
                    "success_lines": success_lines,
                    "total_lines": total_lines,
                    "success_rate": success_rate
                }

                # Print problem-level success rates
                print(f"\n  Problem {problem_idx} Success Rates (by lines of code):")
                print(f"    {llm_type}: {success_lines}/{total_lines} lines passed ({success_rate:.1f}%)")
            
            detailed_results["problems"].append(problem_results)
        
        # Calculate and print overall success rates
        print("\n=== OVERALL SUCCESS RATES ===")
        
        for llm_type, stats in sorted(llm_stats.items()):
            success_rate = (stats["success_lines"] / stats["total_lines"]) * 100 if stats["total_lines"] > 0 else 0
            print(f"{llm_type}: {stats['success_lines']}/{stats['total_lines']} lines passed ({success_rate:.1f}%)")
            
            # Store overall stats for JSON
            detailed_results["overall_stats"][llm_type] = {
                "success_lines": stats["success_lines"],
                "total_lines": stats["total_lines"],
                "success_rate": success_rate
            }
        
        # Find the best performing LLM
        if llm_stats:
            best_llm = max(llm_stats.items(), 
                          key=lambda x: (x[1]["success_lines"] / x[1]["total_lines"]) if x[1]["total_lines"] > 0 else 0)
            best_rate = (best_llm[1]["success_lines"] / best_llm[1]["total_lines"]) * 100 if best_llm[1]["total_lines"] > 0 else 0
            print(f"\nBest performing LLM: {best_llm[0]} ({best_rate:.1f}% of lines passed)")
            
            # Store best performing LLM for JSON
            detailed_results["best_performing_llm"] = {
                "llm_type": best_llm[0],
                "success_rate": best_rate
            }
        
        # Save results to JSON if requested
        if save_to_json:
            import json
            with open(json_path, "w") as f:
                json.dump(detailed_results, f, indent=2)
            print(f"\nDetailed results saved to {json_path}")
