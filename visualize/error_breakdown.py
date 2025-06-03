import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os
import sys

# Add the project root to the path so we can import from core
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.data_classes.llm_type import MODEL_CONFIGS, LLMType

# Matplotlib RC settings for publication (matching pie_chart.py)
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 5,
    'axes.titlesize': 12,
    'axes.labelsize': 10
})

# Load the JSON data
with open('/Users/tianyu/Work/paper2code/outputs/error_classification/2025-05-22-11-31-38/2025-05-22-11-31-39/error_classification_results.json', 'r') as f:
    data = json.load(f)

# Count occurrences of each category by model
error_counts = defaultdict(lambda: defaultdict(int))
models = set()
error_categories = set()  # Set to collect all unique error categories

# The JSON has a different structure than initially expected
# Looking at the actual data format to properly extract info
for item in data:
    if 'llm' in item and 'category_name' in item:
        model = item['llm']
        model = MODEL_CONFIGS[LLMType[model]].pretty_name
        # model_config = MODEL_CONFIGS[LLMType(model)]
        # model_name = model_config.pretty_name
        error_category = item['category_name']
        if error_category:  # Only count if error category is not empty
            error_counts[error_category][model] += 1
            models.add(model)
            error_categories.add(error_category)  # Add to our set of categories

# Sort the error categories for consistency
categories = sorted(list(error_categories))
print(f"Found {len(categories)} unique error categories: {categories}")

# replace Logic Errors with Functional Errors
# breakpoint()
# Define colors for each category
color_map = {
    "Name Errors": "#c1e1c1",       # Light green
    "Logic Errors": "#f9d5a7",      # Light orange
    "Import Errors": "#c6d4e9",     # Light blue
    "Syntax Errors": "#e2ecc6",     # Light yellow-green
    "Index/Key Errors": "#f9e8b8",  # Light yellow
    "Attribute Errors": "#e8d5c8",  # Light brown
    "Type Errors": "#d4c5e2",       # Light purple
}

# Create a figure with an appropriate grid of subplots based on the number of categories
n_categories = len(categories)
n_cols = 4
n_rows = (n_categories + n_cols - 1) // n_cols  # Ceiling division to get enough rows
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]  # Handle single subplot case

# For each category, create a bar chart
for i, category in enumerate(categories):
    ax = axes[i]
    
    # Get the data for this category and sort by count
    if category in error_counts:
        category_data = error_counts[category]
        sorted_models = sorted(category_data.items(), key=lambda x: x[1], reverse=True)
        models_list = [x[0] for x in sorted_models]
        counts = [x[1] for x in sorted_models]
    else:
        models_list = []
        counts = []
    
    # Create the horizontal bar chart
    bars = ax.barh(range(len(models_list)), counts, color=color_map.get(category, "#d3d3d3"))
    
    # Set the y-tick labels to be the model names
    ax.set_yticks(range(len(models_list)))
    ax.set_yticklabels(models_list, fontsize=8)
    
    # Add gridlines
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # Set the title and axis labels
    if category == "Logic Errors":
        ax.set_title("Functional Errors")
    else:
        ax.set_title(category)
    ax.set_xlabel("Count")
    
    # Remove the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Hide any unused subplots
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

# Adjust the layout
plt.tight_layout()

# Save the figure
output_dir = '/Users/tianyu/Work/paper2code/outputs/error_classification/2025-05-22-11-31-38/2025-05-22-11-31-39/error_visualizations'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'error_breakdown.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'error_breakdown.pdf'), dpi=300, bbox_inches='tight')  # Also save as PDF
plt.close()
