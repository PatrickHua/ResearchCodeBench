import re
from typing import Union
import ast
import textwrap

def put_code_blocks_back(code_blocks):
    
    code = ''
    for code_block in code_blocks:
        code += f"```python\n{code_block}\n```\n\n"
    return code
def remove_body_ast(context, function_name, class_name = None, rm_docstring=False) -> Union[str, bool]:
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
        # breakpoint()
        return 'INVALID'

    target_function = None
    for node in ast.walk(module):
        # breakpoint()
        if class_name is not None:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                # loop through all the methods in the class
                for method in node.body:
                    if isinstance(method, ast.FunctionDef) and method.name == function_name:
                        target_function = method
                        
                        break
        else:
            # breakpoint()
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
def remove_body_ast_from_context(code, fn_id, is_method, rm_docstring=False):
    if is_method:
        class_name = fn_id.split(".")[-2]
        function_name = fn_id.split(".")[-1]
        file_name = '/'.join(fn_id.split(".")[:-2]) + '.py'
    else:
        class_name = None
        function_name = fn_id.split(".")[-1]
        file_name = '/'.join(fn_id.split(".")[:-1]) + '.py'

    code_blocks = extract_code_blocks(code)
    new_code_blocks = []
    for code_block in code_blocks:
        if file_name in code_block.split('\n')[0]:
            
            result = remove_body_ast(code_block, function_name, class_name=class_name, rm_docstring=rm_docstring)
            if result == 'INVALID':
                return 'INVALID'
            if result:
                new_code_blocks.append(result)
        else:
            new_code_blocks.append(code_block)

    new_code = put_code_blocks_back(new_code_blocks)
    return new_code




if __name__ == "__main__":
    example = '3'
    with open(f"core/misc/tests/context{example}_fn_id.py", "r") as f:
        fn_id = f.read()
    with open(f"core/misc/tests/context{example}.py", "r") as f:
        context = f.read()
    is_method = False
    


    
    context_gutted = remove_body_ast_from_context(context, fn_id, is_method)
    with open(f"core/misc/tests/context{example}_gutted.py", "w") as f:
        f.write(context_gutted)
