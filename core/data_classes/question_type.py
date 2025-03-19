from enum import Enum


class QuestionType(str, Enum):
    MASKING = "masking"
    FULL_CODE = "full_code"
    CODE_WITH_DOCSTRING = "code_with_docstring"
    CODE_WITH_COMMENTS = "code_with_comments"
    CODE_WITH_COMMENTS_AND_MASK = "code_with_comments_and_mask"
    CODE_WITH_LLM_GEN_SPECS = "code_with_llm_gen_specs"

