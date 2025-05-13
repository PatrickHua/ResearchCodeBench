import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from matplotlib.cm import get_cmap
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

OPEN_MODEL_LINE_WIDTH = 0.7


# Add the parent directory to sys.path to allow for relative imports
sys.path.insert(0, os.path.abspath('..'))

# Load the data from the JSON file
json_path = os.path.join('..', 'outputs', '20llms_greedy', '2025-05-12-17-13-20', 'overall_stats.json')

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
        data = json.load(f)
    return data

def plot_line_rates():
    # Load data
    data = load_data()
    
    # Extract model names, developers, and line rates
    models = []
    line_rates = []
    developers = []
    is_open_list = []
    
    for model_name, model_data in data['overall_scores'].items():
        pretty_name = model_data['llm_cfg']['pretty_name']
        developer = model_data['llm_cfg']['developer']
        line_rate = model_data['line_rates']['mean']
        is_open = model_data['llm_cfg']['is_open']
        
        models.append(pretty_name)
        line_rates.append(line_rate)
        developers.append(developer)
        is_open_list.append(is_open)
    
    # Sort models by line rates in descending order
    sorted_indices = np.argsort(line_rates)[::-1]
    sorted_models = [models[i] for i in sorted_indices]
    sorted_rates = [line_rates[i] for i in sorted_indices]
    sorted_developers = [developers[i] for i in sorted_indices]
    sorted_is_open = [is_open_list[i] for i in sorted_indices]
    
    # Create a mapping of unique developers to colors using shades of blue
    unique_developers = sorted(set(developers))
    num_developers = len(unique_developers)
    
    # Create a custom blue colormap
    blues = LinearSegmentedColormap.from_list(
        'custom_blues', 
        [
            '#E3F2FD',  # Very light blue
            '#BBDEFB',  # Light blue
            '#90CAF9',  # Medium-light blue
            '#64B5F6',  # Medium blue
            '#42A5F5',  # Medium-strong blue
            '#2196F3',  # Strong blue
            '#1E88E5',  # Deeper blue
            '#1976D2',  # Even deeper blue
            '#1565C0',  # Deep blue
            '#0D47A1'   # Very deep blue
        ], 
        N=num_developers
    )
    
    # Generate colors for each developer
    blue_colors = [blues(i/(num_developers-1) if num_developers > 1 else 0) for i in range(num_developers)]
    developer_colors = {dev: blue_colors[i] for i, dev in enumerate(unique_developers)}
    
    # Create the bar chart with a white background
    fig, ax = plt.subplots(figsize=(7, 4), facecolor='white')
    
    # Get colors for each bar based on the developer
    bar_colors = [developer_colors[dev] for dev in sorted_developers]
    
    # Create bars with different edge styles based on whether they're open source
    bars = []
    for i, (rate, color, is_open) in enumerate(zip(sorted_rates, bar_colors, sorted_is_open)):
        if is_open:
            # Open source models have dashed edge
            bar = ax.bar(i, rate, color=color, edgecolor='black', linestyle='--', linewidth=OPEN_MODEL_LINE_WIDTH)
        else:
            # Closed source models have no edge
            bar = ax.bar(i, rate, color=color, edgecolor='none')
        bars.append(bar)
    
    # Add labels and title
    # ax.set_xlabel('Models', fontweight='bold')
    # ax.set_ylabel('Line Rates (%)', fontweight='bold')
    ax.set_title('Paper2Code-bench (Complete)', fontweight='bold')
    
    # Set x-axis ticks with model names
    ax.set_xticks(range(len(sorted_models)))
    
    # Create model labels with only the model names, no developers
    ax.set_xticklabels(sorted_models, rotation=-65, ha='left', fontsize=8, fontweight='bold')
    
    # Remove x-axis tick marks
    ax.tick_params(axis='x', which='both', length=0)
    ax.tick_params(axis='y', which='both', length=0)
    
    # for label in ax.get_xticklabels():
    #     label.set_x(-10)
    # Bring x-axis labels closer to the axis
    plt.setp(ax.get_xticklabels(), y=0.0)
    
    # # Shift x-axis labels slightly to the left
    # for label in ax.get_xticklabels():
    #     pos = label.get_position()
    #     print(pos)
    #     label.set_position((pos[0] - 30, pos[1]))
    
    # Add a legend for the developers
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=developer_colors[dev], label=dev) 
        for dev in unique_developers
    ]
    
    # Add legend elements for open/closed source
    open_patch = Patch(facecolor='white', edgecolor='black', linestyle='--', linewidth=OPEN_MODEL_LINE_WIDTH, label='Open Models')
    closed_patch = Patch(facecolor='white', edgecolor='none', label='Closed Models')
    legend_elements.extend([open_patch, closed_patch])
    
    # Place the legend at the top right
    ax.legend(handles=legend_elements, loc='upper right', ncol=2, fontsize=8, frameon=True)
    
    # Add values on top of each bar
    # for i, v in enumerate(sorted_rates):
    #     ax.text(i, v + 0.8, f'{v:.1f}', ha='center', fontsize=7, fontweight='bold')
    
    # Add grid lines for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines["left"].set_position(("data", 0))
    # ax.spines["bottom"].set_position(("data", 0))
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(-1, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
    
    # Add "Pass@1" and "Models" labels at the end of axes
    color = '#888888'  # Darker gray instead of lightgrey
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
    plt.savefig(f'{output_dir}/model_line_rates.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/model_line_rates.pdf', bbox_inches='tight')
    print(f"Plots saved to {output_dir}/model_line_rates.png and {output_dir}/model_line_rates.pdf")
    
    # Show the plot
    # plt.show()

if __name__ == "__main__":
    plot_line_rates()
