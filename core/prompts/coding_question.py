QUESTION_PROMPT = (
    "You are given a function signature, docstring, and some context from a research codebase.\n\n"
    "--------------------\n"
    "{prob}\n"
    "--------------------\n\n"
    "Complete the body of {fname}, and return the entire {function_or_method}, including the function signature, "
    "formatted within a markdown code block as follows:\n"
    "```python\n"
    "# YOUR CODE HERE\n"
    "```"
)

QUESTION_PROMPT_WITH_PAPER = (
    "You are given a function signature, docstring, and some context from a research codebase.\n\n"
    "Additionally, you are provided with a research paper that may help guide the implementation.\n\n"
    "Paper:\n"
    "--------------------\n"
    "{paper}\n"
    "--------------------\n\n"
    "--------------------\n"
    "{prob}\n"
    "--------------------\n\n"
    "Complete the body of {fname}, and return the entire {function_or_method}, including the function signature, "
    "formatted within a markdown code block as follows:\n"
    "```python\n"
    "# YOUR CODE HERE\n"
    "```"
)