from typing import List
from core.data_classes.llm_type import LLMConfig, LLMType, MODEL_CONFIGS
def get_nearest_knowledge_cutoff(llm_types: List[LLMType]):
    nearest_knowledge_cutoff = None
    for llm_type in llm_types:
        # breakpoint()
        if MODEL_CONFIGS[llm_type].knowledge_cutoff_date is not None:
            if nearest_knowledge_cutoff is None or MODEL_CONFIGS[llm_type].knowledge_cutoff_date >= nearest_knowledge_cutoff:
                nearest_knowledge_cutoff = MODEL_CONFIGS[llm_type].knowledge_cutoff_date
    return nearest_knowledge_cutoff

# args.knowledge_cutoff_date = get_nearest_knowledge_cutoff(args.llm_types)