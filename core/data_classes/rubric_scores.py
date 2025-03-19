from pydantic import BaseModel
from typing import Optional


class RubricScores(BaseModel):
    # total is 100
    functionality: Optional[float] = None
    logic: Optional[float] = None
    semantic_similarity: Optional[float] = None
    code_quality: Optional[float] = None
    
    def calculate_achieved_score(self) -> float:
        """Calculate the total score achieved by the student."""
        if self.functionality is None or self.logic is None or self.semantic_similarity is None or self.code_quality is None:
            return 0
        return self.functionality + self.logic + self.semantic_similarity + self.code_quality

    @property
    def max_score(self):
        """Return the maximum possible score."""
        return 100

    def percentage_score(self):
        """Return the percentage score achieved by the student."""
        return self.calculate_achieved_score() / self.max_score