from main import parse_args
import json
import os
import asyncio
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from core.data_classes.llm_type import LLMType
import pickle
import time
from datetime import datetime
import matplotlib.cm as cm
import argparse
import sys


categories_prompt = """
1.  **Syntax Errors (`SyntaxError`, `IndentationError`)**
    *   **Characteristics:** The generated code violates the basic grammatical rules of the Python language. This includes missing colons, incorrect indentation, mismatched parentheses/brackets/braces, invalid keywords, or malformed statements. These are usually the easiest errors for the Python interpreter to catch immediately.
    *   **Estimated Percentage (Typical):** 10-20% (LLMs have become better at basic syntax, but it still happens, especially with complex structures or nested blocks).
    *   **Potential Causes:** Model struggles with maintaining structural consistency over long code blocks, misinterpreting nesting requirements, minor "hallucinations" in syntax.
    *   **Potential Improvements:** Better training on syntactically diverse code, incorporating linters/syntax checkers during the generation or refinement process, improved attention mechanisms for structure.

2.  **Name Errors (`NameError`)**
    *   **Characteristics:** The code attempts to use a variable, function, or class name that hasn't been defined or assigned a value in the current scope. This often happens when a model forgets a variable it defined earlier, misspells a name, or tries to use a variable before assignment.
    *   **Estimated Percentage (Typical):** 15-25%
    *   **Potential Causes:** Poor state tracking within the model during generation, hallucinating variable names, misunderstanding variable scope, typos introduced during generation.
    *   **Potential Improvements:** Enhanced context windows, improved state management within the model architecture, fine-tuning on code that emphasizes variable declaration and usage patterns.

3.  **Type Errors (`TypeError`)**
    *   **Characteristics:** An operation or function is applied to an object of an inappropriate type. Examples include trying to add a string to an integer, calling `len()` on an integer, or passing the wrong type of argument to a function.
    *   **Estimated Percentage (Typical):** 15-25%
    *   **Potential Causes:** Model misunderstands the expected data types for variables or function parameters/return values, poor inference of types based on context, incorrect assumptions about library function behavior.
    *   **Potential Improvements:** Training data enriched with type hints, incorporating static type checking concepts during training or generation, fine-tuning on specific library usage patterns.

4.  **Attribute Errors (`AttributeError`)**
    *   **Characteristics:** The code tries to access an attribute or method on an object that doesn't possess it (e.g., calling `my_list.appendd()` instead of `my_list.append()`, or trying to access `.text` on an object that isn't a web request response). This often overlaps with TypeErrors, as using the wrong type of object leads to missing attributes.
    *   **Estimated Percentage (Typical):** 10-20%
    *   **Potential Causes:** Misunderstanding of object-oriented principles, incorrect assumptions about the methods/attributes available on objects returned by library functions, typos in method/attribute names.
    *   **Potential Improvements:** Better training on object-oriented programming, fine-tuning on specific library APIs, potentially using retrieval-augmented generation (RAG) to pull in correct API documentation.

5.  **Index/Key Errors (`IndexError`, `KeyError`)**
    *   **Characteristics:** Attempting to access a list/tuple element using an index that is outside the valid range, or trying to access a dictionary value using a key that doesn't exist in the dictionary.
    *   **Estimated Percentage (Typical):** 5-15%
    *   **Potential Causes:** Off-by-one errors in loops, incorrect logic for calculating indices, assuming a key exists in a dictionary without checking, misunderstanding data structures.
    *   **Potential Improvements:** Training on robust code patterns for collection access (e.g., checking bounds, using `.get()` for dictionaries), better logical reasoning capabilities.

6.  **Import Errors (`ImportError`, `ModuleNotFoundError`)**
    *   **Characteristics:** The code fails to import a module or a specific name from a module. This could be due to misspelling the module/name, trying to import a non-existent module, or issues with relative/absolute import paths.
    *   **Estimated Percentage (Typical):** 5-10%
    *   **Potential Causes:** Hallucinating library names, incorrect assumptions about project structure for imports, typos.
    *   **Potential Improvements:** Training on a wider range of projects with different import structures, validating imports against known package lists.

7.  **Logic Errors / Incorrect Output**
    *   **Characteristics:** The code runs without crashing but produces the wrong result or doesn't behave as intended by the prompt. These errors don't raise standard Python exceptions but would fail specific unit tests checking for correctness (e.g., `AssertionError`).
    *   **Estimated Percentage (Typical):** 20-30% (Often the hardest category)
    *   **Potential Causes:** Deep misunderstanding of the problem requirements, flawed algorithmic logic, incorrect handling of edge cases, misinterpretation of mathematical or complex operations.
    *   **Potential Improvements:** Improved reasoning capabilities, training on code with associated unit tests, Reinforcement Learning from Human Feedback (RLHF) focused on functional correctness, step-by-step reasoning during generation (Chain-of-Thought).
8.  **Other**
    *   **Characteristics:** Other errors that don't fit into the above categories.
    *   **Estimated Percentage (Typical):** 0-10%
    *   **Potential Causes:** Model struggles to classify the error, or the error is very rare.
    *   **Potential Improvements:** Training on a wider range of errors, improved classification logic.
"""

# Define category names for analysis
CATEGORIES = [
    "Syntax Errors",
    "Name Errors",
    "Type Errors",
    "Attribute Errors",
    "Index/Key Errors",
    "Import Errors",
    "Logic Errors",
    "Other"
]


async def classify_single_error(error_data, clients, llm_type=LLMType.GPT_4O_MINI):
    """
    Classify a single error using the LLM.
    
    Args:
        error_data: Dictionary containing error information
        clients: AsyncChatClients instance
        llm_type: LLM type to use for classification
    
    Returns:
        Dictionary with the classification result
    """
    # Create prompt for the LLM
    prompt = f"""You are an expert Python programmer tasked with classifying code errors.
    
Below are the categories of errors you should use:
{categories_prompt}

Please classify the following error into ONE of these categories (1-8). 
Respond with ONLY the category number (1-8). No explanation or additional text.

Problem: {error_data['problem']}
Snippet: {error_data['snippet']}
LLM: {error_data['llm']}
Exit Code: {error_data['exit_code']}

Stderr:
```
{error_data['stderr']}
```

Stdout:
```
{error_data['stdout']}
```

Code:
```python
{error_data['code']}
```

Based on the above, classify this error as one of the categories 1-8:
"""

    # Call LLM for classification
    try:
        response = await clients.run(
            llm_type=llm_type,
            user_message=prompt,
            temperature=0,  # Use deterministic output
            num_completions=1,
            return_full_response=True
        )
        
        # Extract and clean response (should just be a number 1-8)
        response_str = response['response_str'][0].strip()
        
        # Handle possible variations in response format
        category_num = None
        
        # Try to extract just the number
        for char in response_str:
            if char.isdigit() and 1 <= int(char) <= 8:
                category_num = int(char)
                break
        
        # If we couldn't find a valid digit, default to "Other"
        if category_num is None:
            category_num = 8  # "Other" category
            
        # Return the classification result
        return {
            "error_id": id(error_data),
            "problem": error_data["problem"],
            "snippet": error_data["snippet"],
            "llm": error_data["llm"],
            "category_num": category_num,
            "category_name": CATEGORIES[category_num-1],
            "exit_code": error_data["exit_code"]
        }
        
    except Exception as e:
        print(f"Error classifying error: {str(e)}")
        # Default to "Other" category on error
        return {
            "error_id": id(error_data),
            "problem": error_data["problem"],
            "snippet": error_data["snippet"],
            "llm": error_data["llm"],
            "category_num": 8,  # "Other" category
            "category_name": "Other",
            "exit_code": error_data["exit_code"],
            "error": str(e)
        }


def collect_errors_from_pset(pset):
    """
    Collect all error information from test results in the problem set.
    
    Args:
        pset: The problem set containing test results
    
    Returns:
        List of error data dictionaries
    """
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
    
    print(f"Collected {len(error_data)} errors from problem set")
    return error_data


async def classify_errors_batch(error_data_list, clients, output_dir, batch_size=20, llm_type=LLMType.GPT_4O_MINI):
    """
    Classify errors in batches to avoid overwhelming the API.
    
    Args:
        error_data_list: List of error data dictionaries
        clients: AsyncChatClients instance
        output_dir: Directory to save results
        batch_size: Number of errors to process in parallel
        llm_type: LLM type to use for classification
    
    Returns:
        List of classification results
    """
    all_results = []
    
    # Create cache directory if it doesn't exist
    cache_dir = os.path.join(output_dir, "error_classification_cache")
    os.makedirs(cache_dir, exist_ok=True)
    breakpoint()
    # Cache file path
    cache_file = os.path.join(cache_dir, "classified_errors.pickle")
    
    # Interim results file (JSON for easier inspection)
    interim_results_file = os.path.join(output_dir, "interim_classification_results.json")
    
    # Load cached results if available
    cached_results = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                cached_results = pickle.load(f)
            print(f"Loaded {len(cached_results)} cached classification results")
        except Exception as e:
            print(f"Error loading cache: {str(e)}")
    
    # Filter out errors that have already been classified
    errors_to_classify = []
    for error_data in error_data_list:
        error_id = id(error_data)
        if error_id in cached_results:
            all_results.append(cached_results[error_id])
        else:
            errors_to_classify.append(error_data)
    
    print(f"Found {len(errors_to_classify)} errors that need classification")
    
    if not errors_to_classify:
        print("No new errors to classify. Using cached results.")
        json_file = os.path.join(output_dir, "error_classification_results.json")
        with open(json_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved {len(all_results)} classification results to {json_file}")
        return all_results
    
    # Process errors in batches
    total_batches = (len(errors_to_classify) + batch_size - 1) // batch_size
    
    # Track timing for estimating completion time
    start_time = time.time()
    batch_times = []
    
    for batch_idx in range(total_batches):
        batch_start_time = time.time()
        
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(errors_to_classify))
        batch = errors_to_classify[start_idx:end_idx]
        
        print(f"\nProcessing batch {batch_idx+1}/{total_batches} ({len(batch)} errors)")
        
        # Display time estimate if we have processed at least one batch
        if batch_idx > 0:
            avg_batch_time = sum(batch_times) / len(batch_times)
            remaining_batches = total_batches - batch_idx
            est_remaining_time = avg_batch_time * remaining_batches
            
            elapsed_time = time.time() - start_time
            est_total_time = elapsed_time + est_remaining_time
            
            print(f"Progress: {batch_idx}/{total_batches} batches completed ({batch_idx/total_batches*100:.1f}%)")
            print(f"Elapsed time: {elapsed_time/60:.1f} minutes")
            print(f"Estimated remaining time: {est_remaining_time/60:.1f} minutes")
            print(f"Estimated total time: {est_total_time/60:.1f} minutes")
            print(f"Estimated completion time: {time.ctime(start_time + est_total_time)}")
        
        # Create tasks for parallel processing
        tasks = [classify_single_error(error_data, clients, llm_type) for error_data in batch]
        
        # Use tqdm to show progress
        progress_bar = tqdm(total=len(tasks), desc="Classifying errors")
        
        # Process each error and update progress
        batch_results = []
        for task in asyncio.as_completed(tasks):
            result = await task
            batch_results.append(result)
            
            # Update cache with new result
            cached_results[result["error_id"]] = result
            
            # Save cache after each result to ensure we don't lose progress
            with open(cache_file, "wb") as f:
                pickle.dump(cached_results, f)
            
            # Update all results
            all_results.append(result)
            
            # Save interim results after each classification
            with open(interim_results_file, "w") as f:
                json.dump(all_results, f, indent=2)
            
            progress_bar.update(1)
        
        progress_bar.close()
        
        # Record batch completion time
        batch_end_time = time.time()
        batch_duration = batch_end_time - batch_start_time
        batch_times.append(batch_duration)
        
        print(f"Batch {batch_idx+1} completed in {batch_duration/60:.1f} minutes")
        print(f"Average time per error: {batch_duration/len(batch):.1f} seconds")
        
        # Save cache after each batch
        with open(cache_file, "wb") as f:
            pickle.dump(cached_results, f)
        
        # Save interim results after each batch
        with open(interim_results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        
        # Short pause between batches to avoid rate limiting
        if batch_idx < total_batches - 1:
            print("Pausing between batches...")
            time.sleep(5)
    
    # Save final results
    with open(cache_file, "wb") as f:
        pickle.dump(cached_results, f)
    
    # Also save as JSON for easier analysis
    json_file = os.path.join(output_dir, "error_classification_results.json")
    with open(json_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Calculate and display total processing time
    total_time = time.time() - start_time
    print(f"\nTotal classification time: {total_time/60:.1f} minutes")
    print(f"Average time per error: {total_time/len(errors_to_classify):.1f} seconds")
    print(f"Saved {len(all_results)} classification results to {json_file}")
    
    return all_results


def analyze_error_classifications(classification_results, output_dir):
    """
    Analyze the error classification results and generate visualizations.
    
    Args:
        classification_results: List of classification results
        output_dir: Directory to save visualizations
    """
    # Create visualizations directory
    viz_dir = os.path.join(output_dir, "error_visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Count errors by category
    category_counts = defaultdict(int)
    for result in classification_results:
        category_counts[result["category_name"]] += 1
    
    # Sort categories by count
    sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Generate pie chart of overall distribution
    generate_pie_chart(sorted_categories, viz_dir)
    
    # Count errors by LLM and category
    llm_category_counts = defaultdict(lambda: defaultdict(int))
    for result in classification_results:
        llm_category_counts[result["llm"]][result["category_name"]] += 1
    
    # Generate histograms for each category
    generate_category_histograms(llm_category_counts, sorted_categories, viz_dir)
    
    # Generate LLM comparison chart
    generate_llm_comparison_chart(llm_category_counts, viz_dir)
    
    print(f"Visualizations saved to {viz_dir}")


def generate_pie_chart(sorted_categories, output_dir):
    """
    Generate a pie chart of error categories.
    
    Args:
        sorted_categories: List of (category, count) tuples
        output_dir: Directory to save the chart
    """
    # Extract data for plotting
    categories = [cat for cat, _ in sorted_categories]
    counts = [count for _, count in sorted_categories]
    total = sum(counts)
    
    # Calculate percentages
    percentages = [(count / total) * 100 for count in counts]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create pie chart with percentage labels
    patches, texts, autotexts = plt.pie(
        counts, 
        labels=categories, 
        autopct='%1.1f%%',
        startangle=90,
        shadow=False,
        explode=[0.05 if i == 0 else 0 for i in range(len(categories))],  # Explode largest slice
        wedgeprops={'edgecolor': 'white', 'linewidth': 1}
    )
    
    # Enhance text properties
    for text in texts:
        text.set_fontsize(12)
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.title('Distribution of Error Categories', fontsize=16, pad=20)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    # Add count data in legend
    legend_labels = [f"{cat} ({count})" for cat, count in sorted_categories]
    plt.legend(legend_labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
    
    # Save the chart
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_category_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()


def generate_category_histograms(llm_category_counts, sorted_categories, output_dir):
    """
    Generate histograms for each error category comparing LLMs.
    
    Args:
        llm_category_counts: Nested dict of counts by LLM and category
        sorted_categories: List of (category, count) tuples in descending order
        output_dir: Directory to save the charts
    """
    # For each category, generate a histogram
    for category_name, _ in sorted_categories:
        # Extract counts for this category across all LLMs
        llms = []
        counts = []
        
        for llm, categories in llm_category_counts.items():
            llms.append(llm)
            counts.append(categories[category_name])
        
        # Sort by count descending
        sorted_data = sorted(zip(llms, counts), key=lambda x: x[1], reverse=True)
        llms = [x[0] for x in sorted_data]
        counts = [x[1] for x in sorted_data]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create horizontal bar chart
        bars = plt.barh(llms, counts, color='skyblue')
        
        # Add count labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                    f"{width:.0f}", ha='left', va='center')
        
        # Add labels and title
        plt.xlabel('Number of Errors', fontsize=12)
        plt.ylabel('LLM Type', fontsize=12)
        plt.title(f'"{category_name}" Errors by LLM Type', fontsize=14)
        
        # Adjust layout and save
        plt.tight_layout()
        safe_category_name = category_name.replace('/', '_').replace(' ', '_')
        plt.savefig(os.path.join(output_dir, f"error_by_llm_{safe_category_name}.png"), dpi=300)
        plt.close()


def generate_llm_comparison_chart(llm_category_counts, output_dir):
    """
    Generate a chart comparing error distribution across LLMs.
    
    Args:
        llm_category_counts: Nested dict of counts by LLM and category
        output_dir: Directory to save the chart
    """
    # Get all unique categories and LLMs
    all_categories = set()
    for llm, categories in llm_category_counts.items():
        all_categories.update(categories.keys())
    
    # Sort categories alphabetically for consistency
    all_categories = sorted(list(all_categories))
    
    # Calculate total errors per LLM for percentage calculation
    llm_totals = {}
    for llm, categories in llm_category_counts.items():
        llm_totals[llm] = sum(categories.values())
    
    # Sort LLMs by total errors descending
    sorted_llms = sorted(llm_totals.items(), key=lambda x: x[1], reverse=True)
    llms = [x[0] for x in sorted_llms]
    
    # Prepare data for stacked bar chart
    data = {}
    for category in all_categories:
        data[category] = []
        for llm in llms:
            # Get percentage of this category for this LLM
            count = llm_category_counts[llm][category]
            percentage = (count / llm_totals[llm]) * 100 if llm_totals[llm] > 0 else 0
            data[category].append(percentage)
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Create stacked bar chart
    bottom = np.zeros(len(llms))
    colors = cm.tab10(np.linspace(0, 1, len(all_categories)))
    
    for i, category in enumerate(all_categories):
        plt.bar(llms, data[category], bottom=bottom, label=category, color=colors[i % len(colors)])
        bottom += np.array(data[category])
    
    # Add labels and title
    plt.xlabel('LLM Type', fontsize=12)
    plt.ylabel('Percentage of Errors', fontsize=12)
    plt.title('Error Category Distribution by LLM Type', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Add total error count to x-axis labels
    xlabels = [f"{llm}\n({llm_totals[llm]} errors)" for llm in llms]
    plt.gca().set_xticklabels(xlabels)
    
    # Add legend
    plt.legend(title="Error Categories", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add percentage labels
    for i, llm in enumerate(llms):
        plt.text(i, 105, f"Total: {llm_totals[llm]}", ha='center', va='bottom', rotation=45)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "llm_error_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()


async def main_async():
    """
    Main asynchronous function to run the error classification process.
    """
    # Parse command-line arguments using the main parser
    import sys
    import os
    
    # Read environment variables for our custom settings
    env_batch_size = os.environ.get('BATCH_SIZE')
    env_resume_dir = os.environ.get('RESUME_DIR')
    env_analyze_only = os.environ.get('ANALYZE_ONLY', '').lower() in ('true', '1', 'yes')
    
    # Create a namespace for our custom arguments
    extra_args = argparse.Namespace(
        analyze_only=env_analyze_only,
        resume_dir=env_resume_dir,
        batch_size=int(env_batch_size) if env_batch_size and env_batch_size.isdigit() else 20
    )
    
    # Print environment variable settings
    print(f"Environment settings:")
    print(f"  BATCH_SIZE: {extra_args.batch_size}")
    print(f"  RESUME_DIR: {extra_args.resume_dir}")
    print(f"  ANALYZE_ONLY: {extra_args.analyze_only}")
    
    pset, llm_types, clients, output_file, args = parse_args()
    
    # Determine output directory
    if extra_args.resume_dir:
        # Resume from specified directory
        output_dir = extra_args.resume_dir
        print(f"Resuming from existing directory: {output_dir}")
    else:
        # Create new timestamped directory
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        base_output_dir = args.output_dir or "outputs/error_analysis"
        output_dir = os.path.join(base_output_dir, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Error classification output will be saved to: {output_dir}")
    
    # Save configuration
    config = {
        "timestamp": datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
        "checkpoint_dir": args.resume_from_ckpt_dir,
        "llm_type_for_classification": LLMType.GPT_4O_MINI.name,
        "contamination_free": args.contamination_free,
        "analyze_only": extra_args.analyze_only,
        "batch_size": extra_args.batch_size
    }
    
    with open(os.path.join(output_dir, "classification_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Check for existing results if in analyze-only mode
    if extra_args.analyze_only:
        results_file = os.path.join(output_dir, "error_classification_results.json")
        if os.path.exists(results_file):
            print(f"Loading existing classification results from {results_file}")
            with open(results_file, "r") as f:
                results = json.load(f)
            print(f"Loaded {len(results)} existing classification results")
            
            # Analyze and visualize results
            analyze_error_classifications(results, output_dir)
            print(f"\nAnalysis complete! Results saved to: {output_dir}")
            return
        else:
            interim_file = os.path.join(output_dir, "interim_classification_results.json")
            if os.path.exists(interim_file):
                print(f"Found interim results at {interim_file}")
                print(f"Using interim results for analysis...")
                with open(interim_file, "r") as f:
                    results = json.load(f)
                print(f"Loaded {len(results)} interim classification results")
                
                # Analyze and visualize results
                analyze_error_classifications(results, output_dir)
                print(f"\nAnalysis complete! Results saved to: {output_dir}")
                return
            else:
                print(f"No existing results found at {results_file} or {interim_file}")
                print("Either run classification first or specify the correct directory")
                return
    
    # Collect errors from problem set
    error_data_list = collect_errors_from_pset(pset)
    
    # Check if we have any errors to classify
    if not error_data_list:
        print("No errors found in the problem set. Exiting.")
        return
    
    print(f"Starting classification of {len(error_data_list)} errors using {LLMType.GPT_4O_MINI.name}...")
    
    # Classify errors
    results = await classify_errors_batch(
        error_data_list, 
        clients, 
        output_dir,
        batch_size=extra_args.batch_size,  # Use specified batch size
        llm_type=LLMType.GPT_4O_MINI  # Use GPT-4O-MINI as specified
    )
    
    # Analyze and visualize results
    analyze_error_classifications(results, output_dir)
    
    # Create a symbolic link to the latest results
    latest_link = os.path.join(os.path.dirname(output_dir), "latest")
    try:
        if os.path.exists(latest_link) and os.path.islink(latest_link):
            os.unlink(latest_link)
        os.symlink(output_dir, latest_link, target_is_directory=True)
        print(f"Created symbolic link to latest results: {latest_link}")
    except Exception as e:
        print(f"Could not create symbolic link: {str(e)}")
    
    print("\nError classification and analysis complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Visualizations saved to: {os.path.join(output_dir, 'error_visualizations')}")
    print(f"To analyze these results later without rerunning classification, use:")
    print(f"  python -m core.analysis.classify_errors --analyze_only --resume_dir {output_dir}")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main_async())

