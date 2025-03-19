# from core.data_class import Function, Question, QuestionType, LLMType, Paper
from core.data_classes.models import Function, Question, QuestionType, LLMType, Paper
from typing import Optional, List
# from core.utils import remove_body_ast, remove_body_ast_from_context
from core.misc.rm_fn_in_context import remove_body_ast_from_context
from core.prompts.spec_gen import SYSTEM_MESSAGE_TESTS, TASK_MESSAGE
# from core.llms import get_llm_response_async, make_messages
from core.generate_solutions import extract_codeblock
from core.async_chat_clients import AsyncChatClients
from core.utils import filter_functions_with_pass_only


def build_problem_with_llm(function: Function, llm_name: LLMType, question_style: QuestionType):
    code = function.code
    prompt = ""
    
    # TODO: Implement this
    
    return Question(question_style=question_style, prompt=prompt, completions=[])


async def build_problem_with_gen_specs(function: Function, llm_name: LLMType, question_style: QuestionType, clients: AsyncChatClients, max_tokens: int, temperature: float):
    code = function.code
    # prompt = ""
    user_prompt = TASK_MESSAGE.format(
        code_snipppet=code,
        # test_code="",
        # argument_types="",
        # output_type="",
        # example_substring="",
        function_name=function.function_id.split(".")[-1],
    )
    
    # TODO: Implement this

    response = await clients.run(
        llm_type=llm_name, 
        user_message=user_prompt, 
        system_message="You are a helpful assistant.", 
        max_tokens=max_tokens,
        temperature=temperature
    )
    response = response[0]
    
    prompt = extract_codeblock(response)
    
    return prompt



async def create_problems(
    function: Function, 
    paper: Paper, 
    clients: AsyncChatClients, 
    llm_name: Optional[LLMType] = None,  
    question_styles: Optional[List[QuestionType]] = [QuestionType.FULL_CODE], 
    with_docstring: bool = False, 
    max_tokens: int = 4096, 
    temperature: float = 0.5
    ):


    code = function.code
    if filter_functions_with_pass_only(code):
        function.valid = False

    if function.questions is None or len(function.questions) == 0:
        function.questions = []
    else:
        existing_question_styles = [question.question_style for question in function.questions]
        question_styles = [question_style for question_style in question_styles if question_style not in existing_question_styles]


    for question_style in question_styles:
        
        if question_style == QuestionType.FULL_CODE:
                    
            context = function.context
            context = remove_body_ast_from_context(context, function.function_id, is_method=function.is_method, rm_docstring=True)
            if context == 'INVALID':
                function.valid = False
            prompt = context
            
        elif question_style == QuestionType.CODE_WITH_DOCSTRING:
            context = function.context
            context = remove_body_ast_from_context(context, function.function_id, is_method=function.is_method, rm_docstring=False)
            if context == 'INVALID':
                function.valid = False
            prompt = context

        function.questions.append(Question(question_style=question_style, prompt=prompt, completions=[]))
    

    
    return function
