from core.data_classes.llm_type import LLMType
from core.async_chat_clients import AsyncChatClients
from core.annotation.models.prediction import Prediction
from typing import List, Dict
from core.annotation.utils.run_inference import run_inference
import asyncio
import logging

async def generate_predictions(masked_code_str: str, paper_tex: str, llm_types: List[LLMType], n_completions: int = 2, predictions: Dict[str, Prediction] = None, clients: AsyncChatClients = None, temperature: float = 0.5, wo_paper: bool = False) -> Dict[str, Prediction]:
    # Initialize or use existing predictions dictionary
    existing_llm_dict = predictions if predictions is not None else {}
    
    # Track tasks for all needed inference runs
    inference_tasks = []
    llm_type_for_task = []
    
    # Determine which LLM types need inference

    for llm_type in llm_types:
        if llm_type.name not in existing_llm_dict:
            # Need to get all completions for this LLM type
            inference_tasks.append(
                run_inference(masked_code_str, paper_tex, llm_type=llm_type, 
                              n_completions=n_completions, clients=clients, temperature=temperature, wo_paper=wo_paper)
            )
            llm_type_for_task.append((llm_type.name, None))  # No existing prediction
        elif len(existing_llm_dict[llm_type.name].completions) < n_completions:
            # Need to get additional completions for this LLM type
            missing_completions = n_completions - len(existing_llm_dict[llm_type.name].completions)
            inference_tasks.append(
                run_inference(masked_code_str, paper_tex, llm_type=llm_type,
                              n_completions=missing_completions, clients=clients, temperature=temperature, wo_paper=wo_paper)
            )
            llm_type_for_task.append((llm_type.name, existing_llm_dict[llm_type.name]))  # Existing prediction to add to
    
    # Run all inference tasks concurrently
    if inference_tasks:
        logging.info(f"Starting batch of {len(inference_tasks)} LLM inference tasks across {len(set(lt for lt, _ in llm_type_for_task))} model types")
        batch_start_time = asyncio.get_event_loop().time()
        
        inference_results = await asyncio.gather(*inference_tasks)
        
        batch_end_time = asyncio.get_event_loop().time()
        batch_duration = batch_end_time - batch_start_time
        logging.info(f"Completed all LLM inference tasks in {batch_duration:.3f}s")
        
        # Process the results
        for (llm_type_name, existing_prediction), result in zip(llm_type_for_task, inference_results):
            if existing_prediction is None:
                # This is a new LLM type
                existing_llm_dict[llm_type_name] = result
            else:
                # Add to existing completions
                existing_prediction.add_completions(result)
    
    return existing_llm_dict
