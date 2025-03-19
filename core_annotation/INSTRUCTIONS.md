# How to annotate a paper's core code.

1. Download the repository (do not clone it. if you do, remove .git folder).
2. Copy the repository to the `pset` folder.
3. inside of that repo, create a yaml file named paper2code.yaml
4. add the following fields:

```yaml
mask_file_path: models/mar.py  # the file where your core code is

context_file_paths:  # these files will be added to the context of the llm. write null if the mask file is self-contained.
  - models/pixelloss.py  # other files that are used in the code
  - util/visualize.py  # other contextual files that are needed for implementing the core code.

test_entry_point: paper2code_test.py  # will run this entry point script to check if the code is correct. paper2code_test.py is a conventional name.
# Use this name unless you have a good reason to change it.


paper_tex: paper2code_paper.tex  # the tex file for the paper.
# name it paper2code_paper.tex unless you have a good reason to change it.
```

4. Create a tex file for the paper.
 - Go to the arxiv website and download the tex file for the paper.
 - If the paper is a single file, just use that file.
 - If the paper is a collection of files, merge them into one file.
   - `latexpand main_file.tex > merged.tex`
   - Check that the merged file is correct.
     - It has abstract, introduction, related work, methods, experiments, conclusion, appendix, etc.
 - Create a file named `paper2code_paper.tex` in the root of the repository.
 - Copy the merged tex file into the created file.


5. Create a requirements.txt file or environment.yaml if it does not exist.

6. Create a test entry point script.
 - Name the file `paper2code_test.py` unless you have a good reason to change it.
 - This script will be used to check if the generated code of the core code is correct.
 - A command `python paper2code_test.py` will be run each time a snippet is generated.
 - unittest is a conventional choice.
 - Test by checking if the output is as expected.
 - AI generated test is ok, as long as it is correct and have good coverage.
 - The test should be able to run with cpu only. If gpu is needed, let me know.


7. Annotate the core code.
    - `pset/DyT/dynamic_tanh.py` is an example of the annotated code.
    - snippets are annotated with `# <paper2code name="snippet_name">` and `# </paper2code name="snippet_name">`
        - The two comments should be at the **same indentation level** of the code.
    - the name should be concise but informative (as a hint for the llm) and unique for each snippet.
    - You should have a variety of snippet sizes.
        - e.g. a single line, a few lines, such as a for loop, a function, etc.
        - Because different snippet sizes have different difficulty to generate.


8. Commit and push the changes in pset folder to the remote repository. Before you do this, check that:
    - the following files are up to date:
        - `requirements.txt`
        - `paper2code_paper.tex`
        - `paper2code_test.py`
        - `paper2code.yaml`
    - delete the unnecessary files, such as `__pycache__`, `.git/`, `.vscode/`, etc.
    - delete the test-irrelevant files. Ideally keep only the necessary files for the test to run.
    - check the size of the repository. It should be as small as possible.
      - a couple of python files shouldn't be too large. Make sure no hidden large files are included.
    - run test again to make sure the change does not break the test.
    - push to a new branch and let me know!




