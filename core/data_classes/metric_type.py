from typing import List, Optional, Dict
from pydantic import BaseModel
from enum import Enum
from pathlib import Path
import os
import json

# Enums
class MetricType(str, Enum):
    EDIT_DISTANCE = "edit_distance"
    EDIT_RATIO = "edit_ratio"
    LCS_RATIO = "lcs_ratio"
    CODEBLEU = "codebleu"
    AST_DISTANCE = "ast_distance"
    COSINE_SIMILARITY = "cosine_similarity"
    BLEU_SCORE = "bleu_score"
    LLM_SCORE = "llm_score"
    MODERNBERT_SCORE = "modernbert_score"
    QWEN_COSINE_SIMILARITY = "qwen_cosine_similarity"
    RUBRIC_SCORE = "rubric_score"
