import json
import yaml
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from collections import defaultdict

with open("outputs/680snippets/results_summary.json", "r") as f:
    data = json.load(f)

results = {}

for problem in data["problems"]:
    problem_name = problem["problem_name"]
    results[problem_name] = {}
    
    for snippet in problem["snippets"]:
        snippet_name = snippet["snippet_name"]
        llm_results = snippet["llm_results"]
        
        # Calculate average success rate across all LLMs
        total_success_rate = 0
        num_llms = len(llm_results)
        
        for llm in llm_results.values():
            total_success_rate += llm["success_rate"]
        
        avg_success_rate = total_success_rate / num_llms
        results[problem_name][snippet_name] = f"{avg_success_rate:.2f}%"

# Convert to YAML and print
yaml_output = yaml.dump(results, sort_keys=False, default_flow_style=False)
print(yaml_output)

# Print total lines for each snippet
print("Snippet Line Counts:")
print("-" * 50)
for problem in data["problems"]:
    print(f"\nProblem: {problem['problem_name']}")
    for snippet in problem["snippets"]:
        print(f"  {snippet['snippet_name']}: {snippet['total_lines']} lines")

# Collect all line counts
line_counts = []
for problem in data["problems"]:
    for snippet in problem["snippets"]:
        line_counts.append(snippet["total_lines"])

# Calculate statistics
mean_lines = np.mean(line_counts)
median_lines = np.median(line_counts)
std_lines = np.std(line_counts)
total_snippets = len(line_counts)

# Print summary statistics
print("\nSnippet Line Count Summary:")
print("-" * 50)
print(f"Total number of snippets: {total_snippets}")
print(f"Mean lines per snippet: {mean_lines:.2f}")
print(f"Median lines per snippet: {median_lines:.2f}")
print(f"Standard deviation: {std_lines:.2f}")
print(f"Minimum lines: {min(line_counts)}")
print(f"Maximum lines: {max(line_counts)}")

# Print distribution of line counts
print("\nLine Count Distribution:")
print("-" * 50)
line_distribution = Counter(line_counts)
for lines, count in sorted(line_distribution.items()):
    percentage = (count / total_snippets) * 100
    print(f"{lines} lines: {count} snippets ({percentage:.1f}%)")

# Create histogram
plt.figure(figsize=(12, 6))
plt.hist(line_counts, bins=30, edgecolor='black')
plt.title('Distribution of Snippet Line Counts (All Problems)')
plt.xlabel('Number of Lines')
plt.ylabel('Number of Snippets')
plt.grid(True, alpha=0.3)

# Add statistics
plt.axvline(mean_lines, color='r', linestyle='dashed', linewidth=1)
plt.axvline(median_lines, color='g', linestyle='dashed', linewidth=1)
plt.legend([f'Mean: {mean_lines:.1f}', f'Median: {median_lines:.1f}'])

# Save the plot
plt.savefig('snippet_line_distribution.png')
print("\nHistogram saved as 'snippet_line_distribution.png'")

# Collect data for scatter plot
success_rates = []

for problem in data["problems"]:
    for snippet in problem["snippets"]:
        line_count = snippet["total_lines"]
        llm_results = snippet["llm_results"]
        
        # Calculate average success rate across all LLMs
        total_success_rate = 0
        num_llms = len(llm_results)
        
        for llm in llm_results.values():
            total_success_rate += llm["success_rate"]
        
        avg_success_rate = total_success_rate / num_llms
        
        success_rates.append(avg_success_rate)

# Create scatter plot
plt.figure(figsize=(12, 6))
plt.scatter(line_counts, success_rates, alpha=0.6)

# Add trend line
z = np.polyfit(line_counts, success_rates, 1)
p = np.poly1d(z)
plt.plot(line_counts, p(line_counts), "r--", alpha=0.8)

# Add labels and title
plt.title('Snippet Size vs Success Rate')
plt.xlabel('Number of Lines')
plt.ylabel('Average Success Rate (%)')
plt.grid(True, alpha=0.3)

# Add correlation coefficient
correlation = np.corrcoef(line_counts, success_rates)[0, 1]
plt.text(0.02, 0.98, f'Correlation: {correlation:.2f}', 
         transform=plt.gca().transAxes, 
         verticalalignment='top')

# Save the plot
plt.savefig('snippet_size_vs_success.png')
print("\nScatter plot saved as 'snippet_size_vs_success.png'")
