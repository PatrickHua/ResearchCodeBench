from pydantic import BaseModel
from typing import List, Optional, Dict
from core_annotation.models.code import Code
from core_annotation.models.prediction import Prediction
from core.data_classes.llm_type import LLMType
from pydantic import Field
from textwrap import dedent
# from core_annotation.models.file import File

class Snippet(BaseModel):
    """
    Represents a single snippet with:
        - name:        string from the tag attribute
        - code:        lines of code *inside* this snippet (no snippet markers)
        - start_line:  index in the file where snippet starts (the marker line)
        - end_line:    index in the file where snippet ends (the marker line)
    """
    name: str
    start_line: int
    end_line: int
    code: Code
    # rel_path: str
    # Make predictions optional with a proper default value
    # predictions: Optional[List[Predictions]] = Field(default_factory=list)
    predictions: Optional[Dict[LLMType, Prediction]] = Field(default_factory=dict)
    

    
    # def update(self, llm_type: LLMType, n_completions: int):
    #     inference_results = run_inference(masked_code_str, repo.paper_tex, llm_types=[llm_type], n_completions=n_completions)
    #     self.predictions.append(inference_results)
        
    def get_indentation(self):
        return self.find_minimum_indentation(self.code)
    
    def __str__(self):
        snippet_str = f"=== Found tag: {self.name} ===\n"
        snippet_str += dedent("".join(self.code.lines))
        return snippet_str

    # def get_non_snippet_lines(self, lines: List[str]) -> List[str]:
    #     """
    #     Returns a list of lines from the file, excluding the snippet.
    #     """
    #     start = self.start_line
    #     end   = self.end_line
    #     return lines[:start], lines[end+1:]



    def find_minimum_indentation(self, code: Code) -> str:
        """
        Returns the minimum common leading whitespace among all non-empty lines.
        If lines are empty or only contain whitespace, we skip them.
        If no non-empty lines exist, returns "" (no indentation).
        """
        indent_levels = []

        for line in code.lines:
            # skip lines that are blank or all whitespace
            if not line.strip():
                continue

            # Count leading spaces/tabs
            leading_whitespace = 0
            for ch in line:
                if ch == ' ' or ch == '\t':
                    leading_whitespace += 1
                else:
                    break
            indent_levels.append(leading_whitespace)

        if not indent_levels:
            return ""  # no non-empty lines => no indentation

        # The minimum indentation is the smallest of these
        min_indent = min(indent_levels)

        # Construct a string of spaces (or tabs) that equals min_indent.
        # Realistically, you might want to preserve exact tabs vs spaces,
        # but for simplicity we just do spaces here:
        return " " * min_indent


    # def test(self, problem_file: ProblemFile):
        