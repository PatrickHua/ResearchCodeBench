from pydantic import BaseModel
from typing import Optional


class CodebleuScores(BaseModel):
    codebleu: Optional[float] = None
    ngram_match_score: Optional[float] = None
    weighted_ngram_match_score: Optional[float] = None
    syntax_match_score: Optional[float] = None
    dataflow_match_score: Optional[float] = None
