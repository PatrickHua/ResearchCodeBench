from core.utils import get_identifier_and_code
import json
import PyPDF2

IDENTIFY_NOVEL_FUNCTION_PROMPT = """
Please identify the top 3 functions from the following list that are most closely related to the core contributions of the paper. 

**Paper Summary:**
- The first {max_page} pages of the paper are provided below. Please read and summarize the method in 100 words.

{paper_text}

**Function List:**
- Below is a list of functions extracted from the repository. Each function is identified by its unique identifier and code.

{function_text}

**Task:**
1. Summarize the method of the paper in 100 words.
2. Identify and list the top 3 function identifiers that are most relevant to the core contributions of the paper.

**Response Format:**
- Provide your response in the following XML-like format:

<summary>
    [Your 100-word summary here]
</summary>

<function_ids>
    <function_id1>function_id1</function_id1>
    <function_id2>function_id2</function_id2>
    <function_id3>function_id3</function_id3>
</function_ids>

Replace function_id1, function_id2, and function_id3 with the actual function identifiers.
"""


SYSTEM_PROMPT = "You are an expert assistant skilled in analyzing research papers and identifying key functions in code repositories that align with the core contributions of the paper."

# TODO: add to the prompt that you should avoid ablations. only core contribution code.
# maybe avoid prompts


def list_functions(functions: list[str]) -> str:

    function_text = ""
    # with open(repo_r2e_parsed_path, 'r') as f:
    #     repo_r2e_parsed = json.load(f)
        # fn_ids = []
        # fn_codes = []
        # for fn in repo_r2e_parsed:
        #     function_text += list_functions(fn)


    function_text = ""
    for fn in functions:
        # fn_id, code = get_identifier_and_code(fn)
        fn_id = fn['function_id']
        code = fn['code']
        function_text += (
            f"Identifier: {fn_id}\n"
            f"Code: \n{code}\n\n"
        )
    return function_text


def get_paper_text(paper_path: str, max_page: int = 8) -> str:
    paper_text = ""
    for page in range(max_page):
        pdf_reader = PyPDF2.PdfReader(paper_path)
        page_text = pdf_reader.pages[page].extract_text()
        paper_text += page_text
    return paper_text