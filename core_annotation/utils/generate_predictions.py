from core.data_classes.llm_type import LLMType
from core.async_chat_clients import AsyncChatClients
from core_annotation.models.prediction import Prediction
from typing import List, Dict
from core_annotation.utils.run_inference import run_inference

async def generate_predictions(masked_code_str: str, paper_tex: str, llm_types: List[LLMType], n_completions: int = 2, predictions: Dict[LLMType, Prediction] = None, clients: AsyncChatClients = None, temperature: float = 0.5) -> Dict[LLMType, Prediction]:
    # TODO: Implement this
    # call llm to generate predictions
    # existing_llm_types = set([prediction.llm_type for prediction in predictions])

    existing_llm_dict = predictions
    for llm_type in llm_types:
        if llm_type not in existing_llm_dict.keys():
            inference_results = await run_inference(masked_code_str, paper_tex, llm_type=llm_type, n_completions=n_completions, clients=clients, temperature=temperature)
            existing_llm_dict[llm_type] = inference_results
        # else:
        if len(existing_llm_dict[llm_type].completions) < n_completions:
            # inference_results = await run_inference(masked_code_str, paper_tex, llm_type=llm_type, n_completions=n_completions-len(existing_llm_dict[llm_type].completions), clients=clients, temperature=temperature)
            inference_results = await run_inference(masked_code_str, paper_tex, llm_type=llm_type, n_completions=n_completions-len(existing_llm_dict[llm_type].completions), clients=clients, temperature=temperature)
            existing_llm_dict[llm_type].add_completions(inference_results)
    
    return existing_llm_dict
