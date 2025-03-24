# How to Annotate a Paper's Core Code

Your folder is located in `./pset/your-repo-name/`. Follow these steps to complete the annotation:

1. **Setup the Environment**
   - Set up your environment and fill out the `paper2code.yaml` file.

2. **Annotate the Core Code**
   - Use `<paper2code name="example"></paper2code name="example">` tags to hierarchically mark key parts of the core function/method.
   - Refer to [`pset/DyT/dynamic_tanh.py`](./pset/DyT/dynamic_tanh.py) for an example.

3. **Complete the Test Script**
   - Ensure `paper2code_test.py` is complete and runs successfully.
   - Complete the `paper2code_requirements.txt` with the minimally needed packages to run `python paper2code_test.py`.
   - Remove files or folders that are not required to run the test.


# Notes on Marking the Core Function
  - Use annotation format:
  ```python
  def sum(numbers: List[Int]) -> Int:
    # <paper2code name="sum a list">
    s = 0
    # <paper2code name="for loop">
    for num in numbers:
      # <paper2code name="add next number">
      s += num
      # <paper2code name="add next number">
    # <paper2code name="for loop">
    return s
    # <paper2code name="sum a list">
  ```

  - Multiple lines can be wrapped in a single <paper2code name="snippet_name">...</paper2code name="snippet_name"> tag with a unique name. The name should be concise but informative (as a hint for the llm).
    - **Function / Method signatures must be placed outside of the tags** to avoid confusion for LLMs.
  - You should have a variety of snippet sizes.
      - e.g. a single line, a few lines, such as a for loop, a function, etc.
      - Because different snippet sizes have different difficulty to generate.
  - Some functions might have branches that are experimental and not of the main interest.
      - Comment out the experimental branches as long as they are not affecting the main logic.
  - **Ensure that the snippets are reproducible solely from the information in the paper.**

## Why are we marking the core function?
>The benchmarking process involves masking key parts of the core function individually, allowing a language model to complete them. The test cases will determine if the generated code functions behave equivalently to the original.
## Why marking hierarchically?
> We are marking the core functions hierarchically to create varying levels of difficulty for LLMs. This allows us to better differentiate their capabilities.

# Notes on writing test cases

  - While we recommend using `unittest` for testing, any format is acceptable as long as it meets these criteria:
    1. Running `python paper2code_test.py` should verify that the generated core code is correct.
    2. The test should raise an error or stop execution if the code is incorrect.
  
  **Important Considerations:**
  - Ensure the test can be executed using only a CPU. If you face difficulties running the test on a CPU, please let us know.
  - A simple testing method is to compare the output against a reference implementation or expected outputs stored in the repository.
  - AI-generated tests are permissible, provided they are accurate and offer comprehensive coverage. We provide an example prompt for you to generate test cases with an AI as a starting point.

```prompt
TBD
```



# Creating a Pull Request

Before committing and pushing to a new branch, verify:

- Files are up to date:
  - `requirements.txt`
  - `paper2code_paper.tex`
  - `paper2code_test.py`
  - `paper2code.yaml`
- Removed unnecessary files:
  - `__pycache__`
  - `.git/`
  - `.vscode/`
  - Other non-essential files
- Repository is minimal:
  - Only test-relevant files remain
  - No hidden large files
  - Overall size is small
- Tests run successfully after changes
- Create a new branch after the name of your repository
- Create PR and request review after pushing