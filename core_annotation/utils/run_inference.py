from core.data_classes.llm_type import LLMType
from core.async_chat_clients import AsyncChatClients
from core_annotation.models.prediction import Prediction, Completion, LLMJudgeResult
from core_annotation.utils.extract_python_block import extract_python_code_blocks
from core_annotation.models.code import Code





async def run_inference(masked_code_str: str, paper_tex: str, llm_type: LLMType, n_completions: int = 2, temperature: float = 0.5, clients: AsyncChatClients = None) -> Prediction:
    prompt = f"""
You are an expert in reproducing research code from a paper.

Here is the paper that you need to use to complete the code:
{paper_tex}

Here is the code that you need to complete:
{masked_code_str}


Please implement the missing code in the TODO blocks. Follow these guidelines carefully:

1. ONLY provide the implementation code that replaces the TODO comments
2. Maintain the exact indentation level of the TODO block
3. Do not include the function/class definitions or any surrounding code
4. Ensure your implementation is complete, functional, and follows best practices
5. Use the paper as a reference for algorithms and techniques

For example, if you see:
def calculate_area(radius):
    # TODO: Implement block "calculate area"
    # Approximately 2 line(s) of code.
    pass

Your answer should include the following:
```python
    area = radius * radius * math.pi
    return area
```
"""
    # with open(f"tmp_prompt.txt", "w") as f:
    #     f.write(prompt)
    # breakpoint()
    completions = await clients.run(
        llm_type=llm_type,
        user_message=prompt,
        num_completions=n_completions,
        temperature=temperature
    )

    with open(f"tmp_completions.txt", "w") as f:
        f.write(str(completions))
    # breakpoint()
    formatted_completions_lines = [Code(extract_python_code_blocks(completion, return_as_list=True)[-1]) for completion in completions]
    
    completions = [Completion(completion=completion, formatted_completion=formatted_completion) for completion, formatted_completion in zip(completions, formatted_completions_lines)]
    return Prediction(completions=completions)

