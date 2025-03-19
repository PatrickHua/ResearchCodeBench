import re
from typing import Union, List

def extract_python_code_blocks(completion: str, return_as_list: bool = False) -> Union[str, List[str]]:
    """
    Extracts content between ```python and ``` tags from the completion string.

    :param completion: The string potentially containing code fences.
    :param return_as_list: If True, return a list of code blocks; otherwise return a single string.
    :return:
        - If return_as_list is False (default), returns the original string if no code blocks are found;
          otherwise returns all code blocks joined with a newline.
        - If return_as_list is True, returns an empty list if no code blocks are found;
          otherwise returns a list of code-block strings.

    Example usage:
        text = "```python\nprint('Hello')\n```\nSome text\n```python\nprint('World')\n```"
        # return_as_list=False => "print('Hello')\nprint('World')"
        # return_as_list=True  => ["print('Hello')", "print('World')"]
    """
    pattern = r'```python\n(.*?)```'
    matches = re.findall(pattern, completion, re.DOTALL)

    if not matches:
        # If no code blocks are found, return the original string or an empty list
        return completion if not return_as_list else []

    # Clean up each matched code block by stripping leading and trailing whitespace
    cleaned_blocks = [code_block for code_block in matches]

    # Return as a list of blocks or join them with a newline
    return cleaned_blocks if return_as_list else "\n".join(cleaned_blocks)