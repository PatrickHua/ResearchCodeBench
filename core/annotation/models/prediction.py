from pydantic import BaseModel
from typing import List
from core.annotation.models.code import Code
from core.data_classes.llm_type import LLMType
from pydantic import BaseModel, Field
from core.annotation.models.code import Code
from typing import Optional



class TestResult(BaseModel):
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    passed: bool

class LLMJudgeResult(BaseModel):
    llm_type: LLMType
    completion: str
    formatted_completion: Code

class Completion(BaseModel):
    completion: str
    formatted_completion: Code
    test_result: Optional[TestResult] = None
    llm_judge_results: Optional[List[LLMJudgeResult]] = None

class Prediction(BaseModel):
    completions: List[Completion] = Field(default_factory=list)
    # formatted_completions_lines: List[Code] = Field(default_factory=list)
    # success: List[bool] = Field(default_factory=list)

    def add_completions(self, predictions: 'Prediction'):
        self.completions.extend(predictions.completions)
        # self.formatted_completions_lines.extend(predictions.formatted_completions_lines)
        # self.success.extend(predictions.success)

    @classmethod
    def create_from_data(cls, llm_type: LLMType, completions: List[str], formatted_completions_lines: List[Code] = None) -> 'Prediction':
        return cls(llm_type=llm_type, completions=completions, formatted_completions_lines=formatted_completions_lines)
