from main import parse_args
import json
import os
from collections import defaultdict
from tqdm import tqdm
from core.data_classes.llm_type import LLMType

def prepare_error_analysis_prompt(pset, output_dir, contamination_free=False):
    """
    Collect all error information from test results and prepare a prompt for GPT-4o
    to identify error patterns and categories.
    
    Args:
        pset: The problem set containing all test results
        output_dir: Directory where to save the prompt
        contamination_free: Whether to use contamination-free results
    
    Returns:
        A prompt string for GPT-4o to analyze
    """
    # Collect all error data
    error_data = []
    
    print("Collecting error information...")
    for problem in tqdm(pset.problems):
        for problem_file in problem.problem_files:
            for snippet in problem_file.snippets:
                for llm_type, predictions in snippet.predictions.items():
                    for completion_idx, completion in enumerate(predictions.completions):
                        if not completion.test_result:
                            continue
                        
                        # Only collect failed test results
                        if not completion.test_result.passed:
                            error_info = {
                                "problem": problem.folder_name,
                                "snippet": snippet.name,
                                "llm": llm_type,
                                "completion_idx": completion_idx,
                                "exit_code": completion.test_result.exit_code,
                                "stdout": completion.test_result.stdout,
                                "stderr": completion.test_result.stderr,
                                "code": str(completion.formatted_completion)
                            }
                            error_data.append(error_info)
    
    # Collect statistics
    total_errors = len(error_data)
    errors_by_llm = defaultdict(int)
    errors_by_problem = defaultdict(int)
    errors_by_exit_code = defaultdict(int)
    
    # Keywords in error messages to track
    common_error_terms = [
        "syntax error", "indentation", "name error", "import", "not defined", 
        "unexpected", "attribute error", "type error", "index error", 
        "key error", "value error", "file not found", "timeout", "memory"
    ]
    error_term_counts = defaultdict(int)
    
    for error in error_data:
        errors_by_llm[error["llm"]] += 1
        errors_by_problem[error["problem"]] += 1
        errors_by_exit_code[str(error["exit_code"])] += 1
        
        # Count error terms in stderr
        error_text = (error["stderr"] or "").lower()
        for term in common_error_terms:
            if term in error_text:
                error_term_counts[term] += 1
    
    # Create a sample of errors to show
    sample_size = min(20, total_errors)
    error_samples = []
    
    # Try to select diverse samples
    llm_samples = {}
    for error in error_data:
        llm = error["llm"]
        if llm not in llm_samples:
            llm_samples[llm] = error
            error_samples.append(error)
            if len(error_samples) >= sample_size:
                break
    
    # If we didn't get enough samples, add more
    if len(error_samples) < sample_size:
        for error in error_data:
            if error not in error_samples:
                error_samples.append(error)
                if len(error_samples) >= sample_size:
                    break
    
    # Create the prompt
    prompt = f"""# Error Pattern Analysis in Code Generation

You are an expert in analyzing patterns in programming errors. I have a dataset of {total_errors} test failures from language models attempting to generate Python code.

## Statistics
- Total errors: {total_errors}
- Errors by LLM type: {dict(errors_by_llm)}
- Errors by exit code: {dict(errors_by_exit_code)}
- Frequency of error terms: {dict(error_term_counts)}

## Your Task
Based on the provided statistics and error samples below, please:

1. Identify the major categories of errors that appear across these failures
2. Describe the characteristics of each error category
3. Estimate what percentage of errors falls into each category
4. Suggest what might be causing these errors and how models could be improved to address them

Don't limit yourself to the predefined error terms - look for deeper patterns across the samples.

## Sample Error Cases
Here are {len(error_samples)} representative error samples from the dataset:

"""

    # Add sample errors
    for i, error in enumerate(error_samples):
        prompt += f"""### Sample {i+1} - Problem: {error['problem']}, LLM: {error['llm']}
Exit Code: {error['exit_code']}

Stderr:
```
{error['stderr']}
```

Stdout:
```
{error['stdout']}
```

Code (partial):
```python
{error['code'][:500]}... (truncated)
```

"""

    # Save the prompt to a file
    filename = f"error_pattern_analysis_{'contamination_free' if contamination_free else ''}.txt"
    with open(os.path.join(output_dir, filename), "w") as f:
        f.write(prompt)
    
    print(f"Saved error analysis prompt to {os.path.join(output_dir, filename)}")
    
    return prompt


async def analyze_error_patterns_with_llm(prompt, clients, output_dir, contamination_free=False, llm_type=None):
    """
    Send the error analysis prompt to the LLM and save the results.
    
    Args:
        prompt: The prepared prompt for error analysis
        clients: The AsyncChatClients instance
        output_dir: Directory to save the results
        contamination_free: Whether the analysis is for contamination-free results
        llm_type: The LLM type to use for analysis (defaults to GPT-4o if None)
    
    Returns:
        The LLM's response
    """
    from core.data_classes.llm_type import LLMType
    
    # Use GPT-4o by default if not specified
    if llm_type is None:
        llm_type = LLMType.GPT_4O
    
    print(f"Sending error analysis prompt to {llm_type.name}...")
    
    system_message = """You are an expert in analyzing programming errors and identifying patterns.
Your task is to analyze a dataset of code generation errors and identify the main categories of errors.
Provide detailed insights into the error patterns you observe."""
    
    try:
        response = await clients.run(
            llm_type=llm_type,
            user_message=prompt,
            system_message=system_message,
            temperature=0,  # Use deterministic output for analysis
            num_completions=1,
            return_full_response=True
        )
        
        # Extract response string
        response_str = response['response_str'][0]
        
        # Save the response to a file
        result_filename = f"error_analysis_results_{'contamination_free' if contamination_free else ''}.txt"
        result_path = os.path.join(output_dir, result_filename)
        
        with open(result_path, "w") as f:
            f.write(response_str)
        
        print(f"Error analysis complete. Results saved to {result_path}")
        print("\n--- Error Analysis Results ---")
        print(response_str[:500] + "..." if len(response_str) > 500 else response_str)
        print("--- End of Preview ---\n")
        
        # Also save the full response object as JSON for reference
        response_data = {
            "analysis": response_str,
            "cost": response['cost'],
            "tokens": {
                "input": clients.total_input_tokens,
                "output": clients.total_output_tokens
            }
        }
        
        json_filename = f"error_analysis_results_{'contamination_free' if contamination_free else ''}.json"
        with open(os.path.join(output_dir, json_filename), "w") as f:
            json.dump(response_data, f, indent=2)
        
        return response_str
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise


async def sanity_check_llm_connection(clients, output_dir, llm_type=None):
    """
    Perform a simple sanity check to verify the LLM connection is working.
    
    Args:
        clients: The AsyncChatClients instance
        output_dir: Directory to save the results
        llm_type: The LLM type to use (defaults to GPT-4o if None)
    
    Returns:
        True if successful, False otherwise
    """
    from core.data_classes.llm_type import LLMType
    
    # Use GPT-4o by default if not specified
    if llm_type is None:
        llm_type = LLMType.GPT_4O
    
    print(f"\n--- LLM Connection Sanity Check ({llm_type.name}) ---")
    
    test_prompt = "Hello! This is a sanity check. Please respond with a simple greeting and confirm you're ready to analyze error patterns."
    
    try:
        response = await clients.run(
            llm_type=llm_type,
            user_message=test_prompt,
            temperature=0,
            num_completions=1,
            return_full_response=True
        )
        
        # Extract response string
        response_str = response['response_str'][0]
        
        # Save the response to a file
        with open(os.path.join(output_dir, "sanity_check_response.txt"), "w") as f:
            f.write(response_str)
        
        print("\nSanity check successful! Response:")
        print("-" * 50)
        print(response_str)
        print("-" * 50)
        print(f"Cost: ${response['cost']:.6f}")
        print(f"Input tokens: {clients.total_input_tokens}")
        print(f"Output tokens: {clients.total_output_tokens}")
        
        return True
        
    except Exception as e:
        print(f"Sanity check failed: {str(e)}")
        return False


if __name__ == "__main__":
    pset, llm_types, clients, output_file, args = parse_args()
    # assert args.resume_from_ckpt_dir is not None
    
    # Load the results from the checkpoint directory
    print(f"Analyzing errors from checkpoint: {args.resume_from_ckpt_dir}")
    
    # Print instructions for the sanity check but also run it directly
    print("\nRunning sanity check of the LLM connection...")
    
    import asyncio
    
    # Run the sanity check
    sanity_check_result = asyncio.run(sanity_check_llm_connection(clients, args.output_dir))
    
    if not sanity_check_result:
        print("Sanity check failed. Please fix the connection issues before proceeding.")
        exit(1)
    
    # Continue with preparing the error analysis prompt
    print("\nSanity check complete. Now preparing the error analysis prompt...")
    
    # Prepare the error analysis prompt
    prompt = prepare_error_analysis_prompt(
        pset=pset, 
        output_dir=args.output_dir,
        contamination_free=args.contamination_free
    )
    
    # Save the prompt
    prompt_file_path = os.path.join(args.output_dir, "error_analysis_prompt.txt")
    with open(prompt_file_path, "w") as f:
        f.write(prompt)
    
    print(f"Saved error analysis prompt to {prompt_file_path}")
    
    # Run the full analysis
    print("\nRunning full error analysis...")
    
    try:
        # Load the prompt from the file to ensure we're using exactly what was saved
        with open(prompt_file_path, "r") as f:
            saved_prompt = f.read()
        
        # Run the analysis
        result = asyncio.run(analyze_error_patterns_with_llm(
            saved_prompt, 
            clients, 
            args.output_dir, 
            args.contamination_free,
            LLMType.GPT_4O_2024_08_06
        ))
        
        print("\nError analysis completed successfully.")
        print("=" * 80)
        print("Full analysis saved to files in the output directory.")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error during full analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nYou can still run the analysis manually with:")
        print("python -c \"import asyncio; from error_analysis import analyze_error_patterns_with_llm; " +
              f"from main import parse_args; pset, llm_types, clients, output_file, args = parse_args(); " +
              f"asyncio.run(analyze_error_patterns_with_llm(open('{prompt_file_path}').read(), " +
              f"clients, '{args.output_dir}', {args.contamination_free}))\"")
    
    