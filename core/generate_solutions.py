# from core.data_class import Repository, Function, Question, Completion, MetricType, LLMType, QuestionType, Paper
from core.data_classes.models import Repository, Function, Question, Completion, Paper
from core.data_classes.llm_type import LLMType
from typing import List, Dict
from core.utils import remove_body_ast_from_context
# from core.prompts import QUESTION_PROMPT, QUESTION_PROMPT_WITH_PAPER
from core.prompts.coding_question import QUESTION_PROMPT, QUESTION_PROMPT_WITH_PAPER
# from core.llms import get_llm_response_async, make_messages
from functools import partial
import ast
import difflib
import argparse
# from visualization_tools.json_visualizer import json2html
import textwrap
from core.utils import run_tasks_in_parallel
import os
import PyPDF2
from core.async_chat_clients import AsyncChatClients

def extract_codeblock(output: str) -> str:
    outputlines = output.split("\n")
    indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if len(indexlines) < 2:
        return ""
    code_block = "\n".join(outputlines[indexlines[0] + 1 : indexlines[1]])
    return textwrap.dedent(code_block)



async def get_completions(function: Function, paper: Paper, clients: AsyncChatClients, llm_names: List[LLMType], max_tokens: int = 4096, temperature: float = 0.5, overwrite_llm_names: List[LLMType] = None, with_paper: bool = False, max_pages: int = 10, max_attempts: int = 1):
    
    # if function.questions is None or len(function.questions) == 0:
    
    for i, question in enumerate(function.questions):

        if overwrite_llm_names is not None:
            for c, completion in enumerate(function.questions[i].completions):
                if completion.llm_name in overwrite_llm_names:

                    function.questions[i].completions.pop(c)

        if function.questions[i].completions is not None and len(function.questions[i].completions) > 0:
            completions = function.questions[i].completions
            existing_llm_names = [completion.llm_name for completion in completions]
            llm_names = [llm_name for llm_name in llm_names if llm_name not in existing_llm_names]
            if len(llm_names) == 0:
                continue

        else:
            completions = []


        # prompt = question.prompt
        # context = function.context
        gutted_context = question.prompt
        # prompt = "Complete the following function: " + prompt + "\n" + context


        if function.is_method:
            class_name = function.function_id.split(".")[-2]
            function_name = function.function_id.split(".")[-1]
            file_name = '/'.join(function.function_id.split(".")[:-2]) + '.py'
            fname = f"`{function_name}` method in the `{class_name}` class from the file `{file_name}`"
            function_or_method = "method"
        else:
            class_name = ''
            function_name = function.function_id.split(".")[-1]
            file_name = '/'.join(function.function_id.split(".")[:-1]) + '.py'
            fname = f"`{function_name}` function from the file `{file_name}`"
            function_or_method = "function"

        if with_paper:
            pdf_reader = PyPDF2.PdfReader(paper.local_path)
            paper_text = ""

            for page in pdf_reader.pages[:min(len(pdf_reader.pages), max_pages)]:
                paper_text += page.extract_text()
            prompt = QUESTION_PROMPT_WITH_PAPER.format(prob=gutted_context, fname=fname, paper=paper_text, function_or_method=function_or_method)
        else:
            prompt = QUESTION_PROMPT.format(prob=gutted_context, fname=fname, function_or_method=function_or_method)


        for llm_name in llm_names:

            for attempt in range(max_attempts):
                _response = await clients.run(
                    llm_type=llm_name, 
                    user_message=prompt, 
                    system_message="You are a helpful assistant.", 
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            
                response = extract_codeblock(_response[0])

                if response == "":
                    # run again
                    continue
                else:
                    break
            # breakpoint()
        
            if response == "":
                breakpoint()
                function.valid = False

            completion = Completion(llm_name=llm_name, completion=response, attempts=attempt+1)
            completions.append(completion)

        function.questions[i].completions = completions

    return function