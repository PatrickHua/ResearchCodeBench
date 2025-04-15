import copy
import os
import shutil

def task_preparation(problem_file, snippet, completion, cache_dir, problem_src_dir, problem_cache_dir, ignore_git):
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