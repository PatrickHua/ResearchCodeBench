import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import matplotlib as mpl
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects

OPEN_MODEL_LINE_WIDTH = 0.7

# Add the parent directory to sys.path to allow for relative imports
sys.path.insert(0, os.path.abspath('..'))

# Load the data from the JSON file
json_path = os.path.join('..', 'outputs', '20llms_greedy', '2025-05-13-10-01-36', 'overall_stats_contamination_free.json')

json_path_complete = os.path.join('..', 'outputs', '20llms_greedy', '2025-05-13-10-01-36', 'overall_stats.json')

# Set up a better visual style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman'] #+ plt.rcParams['font.serif']
# mpl.rcParams['font.family'] = 'sans-serif'
# mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9

# Legend position configuration - adjust these values to position the legend
LEGEND_POSITION_X = 1.0  # Higher values move right
# LEGEND_POSITION_Y = 1.1  # Higher values move up
LEGEND_POSITION_Y = -0.1  # Higher values move up
LEGEND_HORIZ_ALIGN = 'right'  # 'left', 'center', or 'right'
LEGEND_VERT_ALIGN = 'center'  # 'top', 'center', or 'bottom'

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
            # print(model)
            models.append(model)
            rates_2025.append(rates['2025'])
            rates_complete.append(rates['complete'])
    
    # Sort models by 2025+ performance in descending order
    sorted_indices = np.argsort(rates_2025)[::-1]
    sorted_models = [models[i] for i in sorted_indices]
    sorted_rates_2025 = [rates_2025[i] for i in sorted_indices]
    sorted_rates_complete = [rates_complete[i] for i in sorted_indices]
    
    # Reverse the order to put stronger models at the top
    sorted_models = sorted_models[::-1]
    sorted_rates_2025 = sorted_rates_2025[::-1]
    sorted_rates_complete = sorted_rates_complete[::-1]
    
    # Create the bar chart with a white background - now horizontal
    fig, ax = plt.subplots(figsize=(3, 4.1), facecolor='white')
    
    # Add a subtle gradient background
    ax.set_facecolor('#f8f9fa')
    
    # Create a gradient colormap for the bars
    color1 = '#2980b9'  # Deep blue
    color2 = '#3498db'  # Lighter blue
    gradient_cmap = LinearSegmentedColormap.from_list('blue_gradient', [color1, color2], N=256)
    
    # Calculate bar colors based on performance
    normalized_rates = np.array(sorted_rates_2025) / max(sorted_rates_2025)
    bar_colors = [gradient_cmap(rate) for rate in normalized_rates]
    
    # Create the horizontal bars for 2025+ subset with gradient colors
    bars = ax.barh(range(len(sorted_models)), sorted_rates_2025, color=bar_colors, 
                  zorder=2, edgecolor='#ffffff', linewidth=0.5, alpha=0.9)
    
    # Add solid line rectangles for complete benchmark - always on top layer
    bar_height = 0.7  # Default height of bars in matplotlib
    for i, (rate_2025, rate_complete) in enumerate(zip(sorted_rates_2025, sorted_rates_complete)):
        # Draw solid outline for all models, regardless of which score is higher
        rect = Rectangle((0, i - bar_height/2), rate_complete, bar_height, 
                       fill=False, edgecolor='#444444', linestyle='-', linewidth=0.9, zorder=3,
                       alpha=0.7)
        ax.add_patch(rect)
    
    margin = 0.2
    # Reduce margin before first model and after last model
    plt.ylim(-margin-0.6, len(sorted_models) - margin)
    
    # Add title with subtle shadow effect
    # title = ax.set_title('Paper2Code-bench (2025+)', fontweight='bold', fontsize=14, pad=10)
    # title.set_path_effects([path_effects.withStroke(linewidth=2, foreground='#f0f0f0')])
    
    # Set y-axis ticks with model names
    ax.set_yticks(range(len(sorted_models)))
    
    # Create model labels with only the model names - more stylish
    ax.set_yticklabels(sorted_models, ha='right', va='center', fontsize=7, fontweight='bold')
    
    # # Highlight the y-axis labels
    # for label in ax.get_yticklabels():
    #     label.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='#f0f0f0')])
    
    # Make x-axis tick labels smaller and closer to axis
    ax.tick_params(axis='x', which='both', length=0, labelsize=7, pad=1)
    ax.tick_params(axis='y', which='both', length=0, pad=1)
    
    # Add grid lines for better readability
    ax.grid(True, axis='x', linestyle='--', alpha=0.3, color='#555555', zorder=0)
    
    # Remove value annotations at the end of each bar
    # for i, rate in enumerate(sorted_rates_2025):
    #     ax.text(rate + 0.5, i, f'{rate:.1f}', va='center', ha='left', 
    #            fontsize=8, fontweight='bold', color='#505050')
    
    # Add a styled legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='none', edgecolor='#444444', linestyle='-', linewidth=0.9, alpha=0.7, label='ResearchCodeBench'),
        Patch(facecolor=color1, edgecolor='#ffffff', linewidth=0.5, alpha=0.9, label='Contamination-safe Subset')

    ]
    # Create legend with configurable position settings
    legend = ax.legend(handles=legend_elements, 
                      loc=f'{LEGEND_VERT_ALIGN} {LEGEND_HORIZ_ALIGN}',
                      bbox_to_anchor=(LEGEND_POSITION_X, LEGEND_POSITION_Y), 
                      framealpha=0.8,
                      facecolor='white', 
                      edgecolor='#dddddd', 
                      prop={'weight': 'bold', 'size': 9})
    
    # Make all spines visible for a closed box appearance
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    
    # Set spine colors
    ax.spines['top'].set_color('#aaaaaa')
    ax.spines['right'].set_color('#aaaaaa')
    ax.spines['bottom'].set_color('#aaaaaa')
    ax.spines['left'].set_color('#aaaaaa')
    
    # Add "Scaled Pass@1" and "Models" labels with style
    color = '#555555'  # Darker gray
    fontsize = 9
    # subtitle_x = ax.text(1.02, -0.05, "Scaled Pass@1", ha='right', va='center', transform=ax.get_xaxis_transform(), 
    #                     color=color, fontsize=fontsize, fontweight='bold')
    # subtitle_y = ax.text(-0.03, 1.05, "Models", ha='left', va='bottom', transform=ax.get_yaxis_transform(), 
    #                     color=color, fontsize=fontsize, fontweight='bold')
    # subtitle_x.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='#f0f0f0')])
    # subtitle_y.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='#f0f0f0')])
    
    # Add a thin border around the plot
    fig.patch.set_linewidth(0.5)
    fig.patch.set_edgecolor('#dddddd')
    
    # Adjust spacing between y-axis and labels
    plt.subplots_adjust(left=0.25)
    
    # Set x-axis limit to make the plot more readable
    max_rate = max(max(sorted_rates_2025), max(sorted_rates_complete))
    ax.set_xlim(0, max_rate * 1.18)  # Reduce padding from 15% to 5%
    
    # Ensure layout is tight
    plt.tight_layout()
    
    # Create output directory if needed
    output_dir = 'outputs_main_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plots in different formats for publication
    plt.savefig(f'{output_dir}/model_performance_dotted_comparison_ty.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/model_performance_dotted_comparison_ty.pdf', bbox_inches='tight')
    print(f"Plots saved to {output_dir}/model_performance_dotted_comparison_ty.png and {output_dir}/model_performance_dotted_comparison_ty.pdf")
    # plt.show()

if __name__ == "__main__":
    plot_line_rates()
