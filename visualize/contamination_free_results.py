import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import matplotlib as mpl
from matplotlib.patches import Rectangle

OPEN_MODEL_LINE_WIDTH = 0.7

# Add the parent directory to sys.path to allow for relative imports
sys.path.insert(0, os.path.abspath('..'))

# Load the data from the JSON file
json_path = os.path.join('..', 'outputs', '20llms_greedy', '2025-05-13-10-01-36', 'overall_stats_contamination_free.json')

json_path_complete = os.path.join('..', 'outputs', '20llms_greedy', '2025-05-13-10-01-36', 'overall_stats.json')

# Set up a better visual style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9

def load_data():
    with open(json_path, 'r') as f:
        data_2025 = json.load(f)
    
    with open(json_path_complete, 'r') as f:
        data_complete = json.load(f)
        
    return data_2025, data_complete

def plot_line_rates():
    # Load data for both 2025+ subset and complete benchmark
    data_2025, data_complete = load_data()
    
    # Create a dictionary to store model data for both datasets
    model_data = {}
    
    # Extract model names and line rates from 2025+ subset
    for model_name, data in data_2025['overall_scores'].items():
        pretty_name = data['llm_cfg']['pretty_name']
        line_rate_2025 = data['line_rates']['mean']
        
        if pretty_name not in model_data:
            model_data[pretty_name] = {'2025': line_rate_2025}
    
    # Extract line rates from complete benchmark
    for model_name, data in data_complete['overall_scores'].items():
        pretty_name = data['llm_cfg']['pretty_name']
        line_rate_complete = data['line_rates']['mean']
        
        if pretty_name in model_data:
            model_data[pretty_name]['complete'] = line_rate_complete
    
    # Convert to lists for plotting
    models = []
    rates_2025 = []
    rates_complete = []
    
    for model, rates in model_data.items():
        if '2025' in rates and 'complete' in rates:
            models.append(model)
            rates_2025.append(rates['2025'])
            rates_complete.append(rates['complete'])
    
    # Sort models by 2025+ performance in descending order
    sorted_indices = np.argsort(rates_2025)[::-1]
    sorted_models = [models[i] for i in sorted_indices]
    sorted_rates_2025 = [rates_2025[i] for i in sorted_indices]
    sorted_rates_complete = [rates_complete[i] for i in sorted_indices]
    
    # Create the bar chart with a white background
    fig, ax = plt.subplots(figsize=(7, 4), facecolor='white')
    
    # Set light blue color for the 2025+ subset bars
    light_blue = '#ADD8E6'
    
    # Create the bars for 2025+ subset
    bars = ax.bar(range(len(sorted_models)), sorted_rates_2025, color=light_blue, zorder=1)
    
    # Add solid line rectangles for complete benchmark - always on top layer
    bar_width = 0.8  # Default width of bars in matplotlib
    for i, (rate_2025, rate_complete) in enumerate(zip(sorted_rates_2025, sorted_rates_complete)):
        # Draw solid outline for all models, regardless of which score is higher
        rect = Rectangle((i - bar_width/2, 0), bar_width, rate_complete, 
                       fill=False, edgecolor='grey', linestyle='-', linewidth=0.7, zorder=2)
        ax.add_patch(rect)
    
    # Add title
    ax.set_title('Paper2Code-bench (2025+)', fontweight='bold')
    
    # Set x-axis ticks with model names
    ax.set_xticks(range(len(sorted_models)))
    
    # Create model labels with only the model names
    ax.set_xticklabels(sorted_models, rotation=-65, ha='left', fontsize=8, fontweight='bold')
    
    # Remove x-axis tick marks
    ax.tick_params(axis='x', which='both', length=0)
    ax.tick_params(axis='y', which='both', length=0)
    
    # Bring x-axis labels closer to the axis
    plt.setp(ax.get_xticklabels(), y=0.0)
    
    # Add grid lines for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.7, zorder=0)
    
    # Add a simple legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=light_blue, edgecolor='none', label='Paper2Code-bench (2025+)'),
        Patch(facecolor='none', edgecolor='grey', linestyle='-', linewidth=0.7, label='Paper2Code-bench (Complete)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(-1, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
    
    # Add "Pass@1" and "Models" labels at the end of axes
    color = '#888888'  # Darker gray
    fontsize = 8
    ax.text(len(sorted_models)+1, -0.05, "Models", ha='center', va='center', transform=ax.get_xaxis_transform(), color=color, fontsize=fontsize, fontweight='bold')
    ax.text(-0.03, 40.2, "Scaled Pass@1", ha='left', va='bottom', transform=ax.get_yaxis_transform(), color=color, fontsize=fontsize, fontweight='bold')

    # Reduce the gap between y-axis and the leftmost bar
    ax.set_xlim(left=-1, right=len(sorted_models)+0.2)
    
    # Ensure layout is tight
    plt.tight_layout()
    
    # Create output directory if needed
    output_dir = 'outputs_main_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plots in different formats for publication
    plt.savefig(f'{output_dir}/model_performance_dotted_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/model_performance_dotted_comparison.pdf', bbox_inches='tight')
    print(f"Plots saved to {output_dir}/model_performance_dotted_comparison.png and {output_dir}/model_performance_dotted_comparison.pdf")

if __name__ == "__main__":
    plot_line_rates()
