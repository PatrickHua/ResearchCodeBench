# Repository Setup Guide

## Initial Setup

### Create Virtual Environment and Install Requirements

Run these commands in sequence:
```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

### Download GitHub Repo and ArXiv Paper
Run:
```bash
walkthrough.py
```
Follow the prompted steps, including providing the GitHub link and arXiv link for the paper. The program will download the code and paper.

## Repository Configuration

The downloaded repository will be located in `pset/{repo_name}`. Follow these steps to configure it:

> **Note**: Check `pset/DyT` and `pset/fractalgen` for examples of properly configured repositories.

### 1. Requirements Setup
Create a `requirements.txt` file or `environment.yaml` if it doesn't exist to ensure the codebase can be easily run.

### 2. Core Code Identification
Identify the file containing your core code (this will be what the AI is tested on). Update `paper2code.yaml` to include this file path as `mask_file_path`.

### 3. Context Files
Identify files that provide important context for how the codebase and core code works. Update `paper2code.yaml` to include these file paths as `context_file_paths`.

### 4. Test File Creation
Create a new blank file for testing the core code functions. Update `paper2code.yaml` to include this test file path as `test_entry_point`.

### 5. Test File Development
- Create tests to verify the generated core code
- Use `unittest` (recommended)
- Verify outputs match expectations
- AI-generated tests are acceptable if correct and comprehensive
- Tests should run with CPU only (notify if GPU is required)

### 6. Core Code Annotation
- Reference `pset/DyT/dynamic_tanh.py` for annotation examples
- Use annotation format:
  ```python
  # <paper2code name="snippet_name">
  code_here
  # </paper2code name="snippet_name">
  ```
- Guidelines:
  - Comments must be at the **same indentation level** as the code
  - Names should be unique and descriptive
  - Include various snippet sizes (single line, loops, functions, etc.)

## Creating a Pull Request

Before committing and pushing to a new branch, verify:

- [ ] Files are up to date:
  - `requirements.txt`
  - `paper2code_paper.tex`
  - `paper2code_test.py`
  - `paper2code.yaml`
- [ ] Removed unnecessary files:
  - `__pycache__`
  - `.git/`
  - `.vscode/`
  - Other non-essential files
- [ ] Repository is minimal:
  - Only test-relevant files remain
  - No hidden large files
  - Overall size is small
- [ ] Tests run successfully after changes
- [ ] Create PR and request review after pushing