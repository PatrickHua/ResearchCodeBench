import json
import re
import os
import subprocess
import signal
import importlib.util


def dynamical_import(game_path, context=False):
    # dynamically import the 'prompt' from prompt.py under game_path
    if context:
        prompt_path = os.path.join(game_path, "context_prompts.py")
    else:
        prompt_path = os.path.join(game_path, "prompts.py")

    spec = importlib.util.spec_from_file_location("prompt_module", prompt_path)
    prompt_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompt_module)
    iterative_prompts = (
        prompt_module.iterative_prompts
    )  # dynamically import the 'prompt' variable
    try:
        context_prompt_code = (
            prompt_module.context_prompt_code
        )  # dynamically import the 'prompt' variable
    except:
        context_prompt_code = ""
    return iterative_prompts, context_prompt_code


def execute_file(test_file_path, implementation_path):
    result = subprocess.run(
        ["python3", test_file_path, implementation_path], capture_output=True, text=True
    )

    stdout = result.stdout
    stderr = result.stderr
    print(stdout)
    print(stderr)
    # Use regular expressions to find passed tests
    passed_tests = re.findall(r"test_[a-zA-Z_0-9]+ \(.*\) ... ok", stderr)
    passed_tests = [
        re.match(r"(test_[a-zA-Z_0-9]+) \(.*\) ... ok", test).group(1)
        for test in passed_tests
    ]

    # Use regular expressions to find failed tests
    failed_tests = re.findall(r"test_[a-zA-Z_0-9]+ \(.*\) ... FAIL", stderr)
    failed_tests = [
        re.match(r"(test_[a-zA-Z_0-9]+) \(.*\) ... FAIL", test).group(1)
        for test in failed_tests
    ]

    error_tests = re.findall(r"test_[a-zA-Z_0-9]+ \(.*\) ... ERROR", stderr)
    error_tests = [
        re.match(r"(test_[a-zA-Z_0-9]+) \(.*\) ... ERROR", test).group(1)
        for test in error_tests
    ]
    return passed_tests, failed_tests, error_tests


def run_eval(test_file_path, implementation_path, game_name, timeout=20):
    def timeout_handler(signum, frame):
        raise TimeoutError

    json_result = None
    try:
        # Set the signal alarm
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        passed_tests, failed_tests, error_tests = execute_file(
            test_file_path, implementation_path
        )
        with open(f"{game_name}_test_results.json", "r") as f:
            json_result = json.load(f)
        total_tests = len(passed_tests) + len(failed_tests) + len(error_tests)
        # print(f"GPT4 (context) Passed Tests for {game_name}: {passed_tests}")
        # print(f"GPT4 (context) Failed Tests for {game_name}: {failed_tests}")
        # print(f"GPT4 (context) Error Tests for {game_name}: {error_tests}\n")
        if total_tests == 0:
            # couldn't eveen import the file
            acc = 0
        else:
            acc = len(passed_tests) / total_tests
        signal.alarm(0)

    except TimeoutError:
        print("function timed out")
        acc = 0

    return json_result, acc