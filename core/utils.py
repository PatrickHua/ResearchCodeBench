from pebble import concurrent, ProcessPool, ProcessExpired
from typing import Any, Callable, Iterator
import sys
import traceback
import attrs
from enum import Enum
# from tqdm import tqdm
import tqdm
import multiprocessing as mp

def contains_bad_substring(code, bad_code_substrings, occurence=1):
    try:
        # Parse the code into an AST
        tree = ast.parse(code)
        
        # Convert the AST back to code without comments
        code_without_comments = ast.unparse(tree)
        
        # Check for bad substrings in the code without comments
        return sum(bad_code_substring in code_without_comments for bad_code_substring in bad_code_substrings) >= occurence
    except SyntaxError:
        # If the code cannot be parsed, assume it contains bad content
        return True


def get_identifier_and_code(func: dict):
    if func.get('function_id') is not None:
        fn_id = func['function_id']['identifier']
        code = func['function_code']
    elif func.get('method_id') is not None:
        fn_id = func['method_id']['identifier']
        code = func['method_code']
    else:
        raise ValueError(f'Something wrong with {func}')
    return fn_id, code
class TaskRunStatus(Enum):
    SUCCESS = 0
    EXCEPTION = 1
    TIMEOUT = 2
    PROCESS_EXPIRED = 3

@attrs.define(eq=False, repr=False)
class TaskResult:
    status: TaskRunStatus

    result: None | Any = None
    exception_tb: None | str = None

    def is_success(self) -> bool:
        return self.status == TaskRunStatus.SUCCESS

    def is_timeout(self) -> bool:
        return self.status == TaskRunStatus.TIMEOUT

    def is_exception(self) -> bool:
        return self.status == TaskRunStatus.EXCEPTION

    def is_process_expired(self) -> bool:
        return self.status == TaskRunStatus.PROCESS_EXPIRED


def run_tasks_in_parallel_iter(
    func: Callable,
    tasks: list[Any],
    num_workers: int = 2,
    timeout_per_task: None | int = None,
    use_progress_bar: bool = False,
    progress_bar_desc: None | str = None,
    max_tasks_per_worker: None | int = None,
    use_spawn: bool = True,
    max_mem: int = 1024 * 1024 * 1024 * 4,
) -> Iterator[TaskResult]:
    """
    Args:
        func: The function to run. The function must accept a single argument.
        tasks: A list of tasks i.e. arguments to func.
        num_workers: Maximum number of parallel workers.
        timeout_per_task: The timeout, in seconds, to use per task.
        use_progress_bar: Whether to use a progress bar. Default False.
        progress_bar_desc: String to display in the progress bar. Default None.
        max_tasks_per_worker: Maximum number of tasks assigned
        to a single process / worker. None means infinite.
            Use 1 to force a restart.
        use_spawn: The 'spawn' multiprocess context is used. 'fork' otherwise.
    Returns:
        A list of TaskResult objects, one per task.
    """

    mode = "spawn" if use_spawn else "fork"

    with ProcessPool(
        # initializer=initializer if platform.system() != "Darwin" else None,  # type: ignore
        max_workers=num_workers,
        max_tasks=0 if max_tasks_per_worker is None else max_tasks_per_worker,
        context=mp.get_context(mode),
        # initargs=None,#(max_mem,) if platform.system() != "Darwin" else None,  # type: ignore
    ) as pool:
        future = pool.map(func, tasks, timeout=timeout_per_task)

        iterator = future.result()
        if use_progress_bar:
            pbar = tqdm.tqdm(
                desc=progress_bar_desc,
                total=len(tasks),
                dynamic_ncols=True,
            )
        else:
            pbar = None

        succ = timeouts = exceptions = expirations = 0

        while True:
            try:
                result = next(iterator)

            except StopIteration:
                break

            except TimeoutError as error:
                yield TaskResult(
                    status=TaskRunStatus.TIMEOUT,
                )

                timeouts += 1

            except ProcessExpired as error:
                yield TaskResult(
                    status=TaskRunStatus.PROCESS_EXPIRED,
                )
                expirations += 1

            except Exception as error:
                exception_tb = traceback.format_exc()

                yield TaskResult(
                    status=TaskRunStatus.EXCEPTION,
                    exception_tb=exception_tb,
                )
                exceptions += 1

            else:
                yield TaskResult(
                    status=TaskRunStatus.SUCCESS,
                    result=result,
                )

                succ += 1

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    succ=succ, timeouts=timeouts, exc=exceptions, p_exp=expirations
                )
                sys.stdout.flush()
                sys.stderr.flush()



def run_tasks_in_parallel(
    func: Callable,
    tasks: list[Any],
    num_workers: int = 2,
    timeout_per_task: None | int = None,
    use_progress_bar: bool = False,
    progress_bar_desc: None | str = None,
    max_tasks_per_worker: None | int = None,
    use_spawn: bool = True,
) -> list[TaskResult]:
    """
    Args:
        func: The function to run. The function must accept a single argument.
        tasks: A list of tasks i.e. arguments to func.
        num_workers: Maximum number of parallel workers.
        timeout_per_task: The timeout, in seconds, to use per task.
        use_progress_bar: Whether to use a progress bar. Defaults False.
        progress_bar_desc: String to display in the progress bar. Default None.
        max_tasks_per_worker: Maximum number of tasks assigned to a single
        process / worker. None means infinite.
            Use 1 to force a restart.
        use_spawn: The 'spawn' multiprocess context is used. 'fork' otherwise.
    Returns:
        A list of TaskResult objects, one per task.
    """

    task_results: list[TaskResult] = list(
        run_tasks_in_parallel_iter(
            func=func,
            tasks=tasks,
            num_workers=num_workers,
            timeout_per_task=timeout_per_task,
            use_progress_bar=use_progress_bar,
            progress_bar_desc=progress_bar_desc,
            max_tasks_per_worker=max_tasks_per_worker,
            use_spawn=use_spawn,
        )
    )

    return task_results

def remove_body(context, function_name):
    
    # deprecated
    print("remove_body is deprecated. Use remove_body_ast instead.")
    import re

    def replacement(match):
        signature = match.group(1)
        docstring = match.group(2) or ""
        return f"{signature}{docstring}pass\n"

    pattern = rf'(def\s+{re.escape(function_name)}\s*\([^)]*\):)(\s*"""[\s\S]*?"""\s*)?([^@]+)'
    modified_context = re.sub(pattern, replacement, context, flags=re.DOTALL)
    return modified_context

import ast
import textwrap

def extract_body_ast(context, function_name):
    # Remove comments and dedent the context.
    context_no_comments = "\n".join(
        line for line in context.splitlines() 
        if line.strip() and not line.lstrip().startswith("#")
    )
    dedented_context = textwrap.dedent(context_no_comments)

    def ast_node_to_code(node: ast.AST) -> str:
        """
        Converts an AST node back into valid Python source code.
        """
        try:
            # For Python 3.9+:
            return ast.unparse(node)
        except AttributeError:
            # Fallback for Python versions without ast.unparse:
            import astor
            return astor.to_source(node)

    try:
        module = ast.parse(dedented_context)
        for node in ast.walk(module):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                body_nodes = node.body

                # If the first statement is a docstring, remove it.
                if (body_nodes and isinstance(body_nodes[0], ast.Expr) and 
                    isinstance(body_nodes[0].value, ast.Constant)):
                    # For Python 3.8+ ast.Constant is used.
                    if isinstance(body_nodes[0].value, ast.Constant):
                        if isinstance(body_nodes[0].value.value, str):
                            body_nodes = body_nodes[1:]
                    else:
                        # For older Python versions using ast.Str.
                        body_nodes = body_nodes[1:]

                # Option 1: Convert each statement individually and join.
                code = "\n".join(ast_node_to_code(stmt) for stmt in body_nodes)
                
                # Option 2 (alternative): Wrap the statements in a new Module.
                # new_module = ast.Module(body=body_nodes, type_ignores=[])
                # code = ast_node_to_code(new_module)
                
                return code

    except Exception as e:
        # Optionally handle exceptions or debug.
        # breakpoint()
        
        return ""

    return ""
import ast

def filter_functions_with_pass_only(code):
    """
    Filters out functions that only contain a 'pass' statement.
    
    :param code: A string containing the Python code to analyze.
    :return: True if the code contains only a 'pass' statement, False otherwise.
    """
    # Parse the code into an AST
    try:
        module = ast.parse(code)
    except Exception as e:
        # breakpoint()
        return False
    
    # List to store function names that do not contain only 'pass'
    valid_functions = []
    
    # Traverse the AST to find function definitions
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef):
            # Check if the function body contains only a 'pass' statement
            if not (len(node.body) == 1 and isinstance(node.body[0], ast.Pass)):
                valid_functions.append(node.name)
    
    return len(valid_functions) == 0 # if no pass functions, return True

def extract_function_without_comments(context, function_name):
    # Remove comments and dedent the context.
    context_no_comments = "\n".join(
        line for line in context.splitlines() 
        if line.strip() and not line.lstrip().startswith("#")
    )
    dedented_context = textwrap.dedent(context_no_comments)

    def ast_node_to_code(node: ast.AST) -> str:
        """
        Converts an AST node back into valid Python source code.
        """
        try:
            # For Python 3.9+:
            return ast.unparse(node)
        except AttributeError:
            # Fallback for Python versions without ast.unparse:
            import astor
            return astor.to_source(node)

    try:
        module = ast.parse(dedented_context)
        for node in ast.walk(module):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                # Remove the docstring if present
                if (node.body and isinstance(node.body[0], ast.Expr) and 
                    isinstance(node.body[0].value, ast.Constant) and 
                    isinstance(node.body[0].value.value, str)):
                    node.body = node.body[1:]

                # Convert the entire function back to code
                code = ast_node_to_code(node)
                return code

    except Exception as e:
        # Optionally handle exceptions or debug.
        # breakpoint()
        
        return ""

    return ""
import ast
import textwrap
from typing import Union
import re

def remove_body_ast(context, function_name, rm_docstring=False) -> Union[str, bool]:
    """
    Removes the body (and any comments within it) of the specified function and
    replaces it with a single 'pass' statement, optionally preserving the docstring.
    All other comments and code outside remain intact.
    
    Args:
        context (str): The full source code.
        function_name (str): The name of the function to modify.
        rm_docstring (bool): If True, also remove the docstring. Defaults to False.
    
    Returns:
        str: The modified source code, or False if no matching function was found.
    """
    # For robust parsing, dedent the entire source.
    dedented_context = textwrap.dedent(context)
    
    try:
        module = ast.parse(dedented_context)
    except Exception:
        # If parsing fails, return the original context.
        return context

    target_function = None
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            target_function = node
            break

    if target_function is None:
        return False

    # Build a new function body.
    new_body = []
    # Optionally preserve the docstring (which is an Expr node as the first statement).
    if not rm_docstring and target_function.body:
        first_stmt = target_function.body[0]
        if (isinstance(first_stmt, ast.Expr) and 
            (isinstance(first_stmt.value, ast.Str) or
             (hasattr(first_stmt.value, "s") and isinstance(first_stmt.value.s, str)) or
             (isinstance(first_stmt.value, ast.Constant) and isinstance(first_stmt.value.value, str)))):
            new_body.append(first_stmt)
    # Replace the rest of the body with a single pass.
    new_body.append(ast.Pass())
    target_function.body = new_body

    # Generate new source code for just this function.
    try:
        new_func_source = ast.unparse(target_function)
    except AttributeError:
        import astor
        new_func_source = astor.to_source(target_function)

    # --- Reindent the new function source ---
    #
    # The AST-generated source comes out flush left, but the original function may have
    # been indented (e.g. inside a class). We recover the original function’s indent
    # from the dedented source.
    dedented_lines = dedented_context.splitlines()
    # AST line numbers are 1-indexed.
    start_idx = target_function.lineno - 1
    # Get the indent of the function definition line from the dedented source.
    func_line = dedented_lines[start_idx] if start_idx < len(dedented_lines) else ""
    m = re.match(r"^(\s*)", func_line)
    func_indent = m.group(1) if m else ""
    
    # Reindent each line of the new function source with the recovered indent.
    new_func_lines = new_func_source.splitlines()
    reindented_new_func_lines = [
        func_indent + line if line.strip() != "" else line
        for line in new_func_lines
    ]
    new_func_source_reindented = "\n".join(reindented_new_func_lines)
    
    # Determine where the function ends. (Python 3.8+ provides end_lineno.)
    if hasattr(target_function, 'end_lineno') and target_function.end_lineno is not None:
        end_idx = target_function.end_lineno  # end_idx is exclusive when slicing.
    else:
        end_idx = start_idx + 1  # Fallback if end_lineno isn’t available.
    
    # Replace the original function region in the dedented source.
    new_dedented_lines = (
        dedented_lines[:start_idx] +
        new_func_source_reindented.splitlines() +
        dedented_lines[end_idx:]
    )
    new_dedented_source = "\n".join(new_dedented_lines)
    
    # --- Reapply the original file's common indent if any ---
    #
    # Sometimes the original context was indented (e.g. inside a triple-quoted string).
    # Compute the common indent of all non-blank lines in the original context and reapply it.
    original_lines = context.splitlines()
    def common_indent(lines):
        indents = []
        for line in lines:
            if line.strip():
                m = re.match(r"^(\s*)", line)
                if m:
                    indents.append(m.group(1))
        if not indents:
            return ""
        # Use the shortest indent as the common indent.
        return min(indents, key=len)
    orig_common = common_indent(original_lines)
    
    if orig_common:
        final_lines = [
            orig_common + line if line.strip() != "" else line
            for line in new_dedented_source.splitlines()
        ]
        final_source = "\n".join(final_lines)
    else:
        final_source = new_dedented_source
    
    return final_source

def extract_code_blocks(text):
    """
    Extracts code blocks from a string that are enclosed by triple backticks with a 'python' tag.

    Args:
        text (str): The input string containing code blocks.

    Returns:
        list: A list of strings, each containing the code from one block.
    """
    # The regex pattern explanation:
    # - ```python matches the literal string "```python"
    # - \s* matches any whitespace (including newlines) after the language tag
    # - (.*?) lazily captures everything (including newlines) as the code block content
    # - \s*``` matches any trailing whitespace followed by the closing triple backticks
    pattern = r"```python\s*(.*?)\s*```"
    
    # Use re.DOTALL so that the '.' character matches newlines as well
    code_blocks = re.findall(pattern, text, re.DOTALL)
    
    return code_blocks

def put_code_blocks_back(code_blocks):
    
    code = ''
    for code_block in code_blocks:
        code += f"```python\n{code_block}\n```\n\n"
    return code


def remove_body_ast_from_context(code, function_name, rm_docstring=False):

    code_blocks = extract_code_blocks(code)

    new_code_blocks = []
    for code_block in code_blocks:
        result = remove_body_ast(code_block, function_name, rm_docstring=rm_docstring)
        if result:
            new_code_blocks.append(result)
        else:
            new_code_blocks.append(code_block)

    new_code = put_code_blocks_back(new_code_blocks)
    return new_code


from core.data_classes.models import Repository, Function, Question
from core.data_classes.question_type import QuestionType
from core.data_classes.llm_type import LLMType
from core.load_repo import init_repo
from typing import List, Callable, Union, Optional, Tuple
from functools import partial
import ast
import re
from core.load_repo import save_repos
# from visualization_tools.json_visualizer import json2html
from core.utils import run_tasks_in_parallel, remove_body
import asyncio
from tqdm.asyncio import tqdm


def parse_grade(grade_text, patterns=['student_points', 'total_points'], exclude_patterns=[]):
    matches = []
    if len(exclude_patterns) > 0:
        for pattern in exclude_patterns:
            grade_text = re.sub(f'<{pattern}>(.*?)</{pattern}>', '', grade_text)

    for pattern in patterns:
        match = re.search(f'<{pattern}>(.*?)</{pattern}>', grade_text)
        if match:
            matches.append(match.group(1))
        else:   
            raise ValueError(f"Pattern {pattern} not found in grade_text")
    return matches


async def update_repo_functions(repo: Repository, filter_fn: Callable[[Function], bool] = lambda x: True, update_fn: Optional[Callable[[Function], Function]] = None):
    # Create a new list of functions that pass the filter
    updated_functions = []
    for function in repo.functions:
        if filter_fn(function) and function.valid:
            updated_functions.append(update_fn(function, repo.paper))
        # If the function does not pass the filter, it is not added to the updated list
    updated_functions = await asyncio.gather(*updated_functions)
    # Update the repo's functions with the filtered list
    repo.functions = updated_functions
    # breakpoint()
    # print(len(repo.functions))
    return repo


async def update_functions_for_all_repos(
    repos: List[Repository],
    filter_type: str = "all",
    update_fn: Optional[Callable[[Function], Function]] = None,
    num_workers: int = 10, 
    use_progress_bar: bool = True, 
    progress_bar_desc: str = "Updating functions"):

    updated_repos = []
    tasks = []

    for repo in repos:
        filter_fn = get_filter_function(repo, filter_type)
        task = update_repo_functions(repo, filter_fn, update_fn)
        tasks.append(task)

    if use_progress_bar:
        # Use tqdm to wrap the asyncio.gather call
        updated_repos = await tqdm.gather(*tasks, desc=progress_bar_desc, total=len(repos))
    else:
        updated_repos = await asyncio.gather(*tasks)

    return updated_repos


def filter_function_by_id(function_ids: List[str], function: Function) -> bool:
    if function.function_id in function_ids:
        return True
    else:
        return False


def get_filter_function(repo: Repository, filter_type: Optional[str]) -> Callable[[Function], bool]:
    if filter_type == "novel":
        if repo.novel_fn_ids is not None and len(repo.novel_fn_ids) > 0:
            # print(f"Filtering functions by id: {repo.novel_fn_ids}")
            return partial(filter_function_by_id, repo.novel_fn_ids)
        else:
            return lambda x: True
    elif filter_type is None:
        # print("No filter")
        return lambda x: True
    else:
        raise ValueError(f"Invalid filter type: {filter_type}")



