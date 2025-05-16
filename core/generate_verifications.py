# from core.load_repo import init_repo

# from core.data_class import Repository, Function, Question, Completion, MetricType, LLMType, QuestionType, Paper
from core.data_classes.models import Repository, Function, Question, Completion, Paper
from core.data_classes.metric_type import MetricType
from core.data_classes.llm_type import LLMType
from typing import List
from core.utils import remove_body
# from core.llms import get_llm_response, make_messages, get_llm_response_async
from core.prompts.rubric import GRADING_PROMPT
from codebleu import calc_codebleu
import editdistance
import Levenshtein
import pylcs

import torch
from core.embedding_models.embedding_models import forward_qwen
from core.embedding_models.modern_bert import cosine_similarity_cls, forward_modernbert
from core.utils import parse_grade
from core.data_classes.rubric_scores import RubricScores
from core.utils import extract_function_without_comments
from core.data_classes.codebleu_scores import CodebleuScores
from core.async_chat_clients import AsyncChatClients
# import torch._dynamo

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"


async def get_verifications(function: Function, paper: Paper, clients: AsyncChatClients, llm_name: LLMType, metrics: List[MetricType], max_tokens: int = 4096, temperature: float = 0.5, overwrite_metrics: List[MetricType] = None):
    for i, question in enumerate(function.questions):
        
        for j, completion in enumerate(question.completions):
            for metric in metrics:
                
                
                if function.questions[i].completions[j].metrics is None:
                    function.questions[i].completions[j].metrics = {}
                if metric not in function.questions[i].completions[j].metrics or metric in overwrite_metrics:
                    # breakpoint()
                    completion_body = extract_function_without_comments(completion.completion, function.function_id.split(".")[-1])
                    function_body = extract_function_without_comments(function.code, function.function_id.split(".")[-1])
                    
                    if completion_body == "" or function_body == "":
                        function.questions[i].completions[j].metrics[metric] = 0
                        continue

                    if metric == MetricType.EDIT_DISTANCE:  # lower more similar
                        edit_distance = editdistance.eval(function_body, completion_body)
                        function.questions[i].completions[j].metrics[metric] = edit_distance
                    elif metric == MetricType.EDIT_RATIO:  # higher more similar
                        edit_ratio = Levenshtein.ratio(function_body, completion_body)
                        function.questions[i].completions[j].metrics[metric] = edit_ratio
                        
                    elif metric == MetricType.LCS_RATIO: # higher more similar, longest common substring
                        lcs_distance = pylcs.lcs(function_body, completion_body)
                        # breakpoint()
                        function.questions[i].completions[j].metrics[metric] = lcs_distance / len(function_body)
                    elif metric == MetricType.CODEBLEU:
                        codebleu_score = calc_codebleu([function_body], [completion_body], lang="python", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
                        function.questions[i].completions[j].metrics[metric] = codebleu_score['codebleu']
                        function.questions[i].completions[j].codebleu = CodebleuScores(**codebleu_score)
                        if function.questions[i].completions[j].codebleu.dataflow_match_score == 0:
                            # breakpoint()
                            print(f'{function.function_id} is not valid. dataflow_match_score is 0')
                            function.valid = False
                            
                    elif metric == MetricType.AST_DISTANCE:
                        pass
                    elif metric == MetricType.COSINE_SIMILARITY:
                        pass
                    elif metric == MetricType.BLEU_SCORE:
                        pass
                    elif metric == MetricType.LLM_SCORE:
                        pass
                    elif metric == MetricType.MODERNBERT_SCORE:
                        embedding1, embedding2 = forward_modernbert(function_body, device), forward_modernbert(completion_body, device)
                        cosine_sim = cosine_similarity_cls(embedding1[0], embedding2[0])
                        function.questions[i].completions[j].metrics[metric] = cosine_sim.item()

                    elif metric == MetricType.QWEN_COSINE_SIMILARITY:  # higher more similar
                        try:
                            if len(completion.completion) < len(function.code) // 4:
                                function.questions[i].completions[j].metrics[metric] = 0
                            else:
                                logits1 = forward_qwen(function_body, device)
                                logits2 = forward_qwen(completion_body, device)
                                cosine_sim = torch.nn.functional.cosine_similarity(logits1, logits2)
                                function.questions[i].completions[j].metrics[metric] = cosine_sim.item()
                        except Exception as e:
                            breakpoint()
                            
                    elif metric == MetricType.RUBRIC_SCORE:
                        prompt = GRADING_PROMPT.format(reference_code=function_body, student_code=completion_body)
                        # messages = make_messages(system_prompt="You are a helpful assistant.", user_prompt=prompt, model=llm_name)
                        # grading = await get_llm_response_async(messages, model=llm_name)
                        grading = await clients.run(
                            llm_type=llm_name, 
                            user_message=prompt, 
                            system_message="You are a helpful assistant.", 
                            max_tokens=max_tokens,
                            temperature=temperature
                        )
                        grading = grading[0]
                        
                        # Dynamically get the field names from the RubricScores data class, excluding those with default values
                        categories = RubricScores.__annotations__.keys()

                        # Parse the grading using the derived categories
                        try:
                            scores = parse_grade(grading, patterns=categories)
                        except Exception as e:
                            # breakpoint()
                            scores = [None] * len(categories)
                        for s, score in enumerate(scores):
                            if isinstance(score, str):
                                # try to convert to float
                                try:
                                    scores[s] = float(score)
                                except ValueError:
                                    scores[s] = None

                        # Create a RubricScores instance using the parsed scores
                        try:
                            function.questions[i].completions[j].rubric_scores = RubricScores(**dict(zip(categories, scores)))
                            function.questions[i].completions[j].metrics[metric] = function.questions[i].completions[j].rubric_scores.percentage_score()
                        except Exception as e:
                            breakpoint()
        # prompt = question.prompt
        # context = function.context
        # prompt = "Complete the following function: " + prompt + "\n" + context
        # context = remove_body(context, function.function_id.split(".")[-1])
        

    return function