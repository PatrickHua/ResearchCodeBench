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
    #     '#FFF5A5',  # Pineapple ice
    # '#FFD36E',  # Golden melon
    # '#FFA69E',  # Lychee pink
    # '#FF686B',  # Pomegranate fizz
    # '#C94C4C',  # Blood orange
    # '#8E3B46',  # Plum wine
    # '#5B3758',  # Blackberry
    # '#3C2A4D',  # Night fig
    # '#92E6E6',  # Shaved ice
    # '#DFF3E3'   # Aloe mist
    #     '#FFE66D',  # Sunburst
    # '#FF6B6B',  # Flamingo pink
    # '#F65C78',  # Hibiscus
    # '#6A0572',  # Orchid bloom
    # '#374785',  # Indigo dusk
    # '#24305E',  # Navy fig
    # '#3AC1A7',  # Turquoise wave
    # '#30BCED',  # Caribbean sky
    # '#ACE7EF',  # Glacier mist
    # '#F8FFE5'   # Palm dew
    
    
    # '#FFF9F0',  # Eggshell
    # '#FFECD5',  # Vanilla sugar
    # '#FFE0B2',  # Caramel drizzle
    # '#FFD180',  # Honeycomb
    # '#FFC04C',  # Toffee
    # '#FFB300',  # Golden syrup
    # '#F99F0B',  # Toasted sugar
    # '#DD8600',  # Burnt butter
    # '#B56C00',  # Maple
    # '#8E5100'   # Dark molasses
    
    
    
    
    # '#FDF0F0',  # Whisper pink
    # '#FADADD',  # Cherry blossom
    # '#F9C5C9',  # Pastel rose
    # '#F7A1A1',  # Pink petal
    # '#F77F8C',  # Soft ruby
    # '#E46D79',  # Rosé
    # '#C96477',  # Dusty raspberry
    # '#A35D6A',  # Vintage mauve
    # '#844655',  # Cocoa pink
    # '#5C3A3A'   # Earthy rosewood



    # '#FFF1E0',  # Vanilla cream
    # '#FFE0C7',  # Light peach
    # '#FFC7B2',  # Apricot
    # '#FFB3A7',  # Blush
    # '#FF968A',  # Coral
    # '#F97C72',  # Watermelon
    # '#EC5D57',  # Rosewood
    # '#D1495B',  # Pomegranate
    # '#AC3B61',  # Cranberry
    # '#7C2F51'   # Mulberry
    
    #     '#F9ED69',  # Pineapple yellow
    # '#F08A5D',  # Papaya
    # '#B83B5E',  # Watermelon pink
    # '#6A0572',  # Dragonfruit purple
    # '#3E1E68',  # Grape
    # '#2B2D42',  # Shadow navy
    # '#3FEEE6',  # Mint splash
    # '#55BCC9',  # Sky aqua
    # '#97CAEF',  # Tropical blue
    # '#E4F9F5'   # Coconut mist
    # '#F9ED69',  # Pineapple yellow
    # '#F08A5D',  # Papaya
    # '#EB6A80',  # Lighter watermelon pink
    # '#A55CA5',  # Lighter dragonfruit purple
    # '#71589C',  # Lightened grape
    # '#4A4E7B',  # Soft navy (less shadow)
    # '#3FEEE6',  # Mint splash
    # '#66D3E9',  # Softer sky aqua
    # '#A9D8F7',  # Bright tropical blue
    # '#E4F9F5'   # Coconut mist
    
    # '#F9ED69',  # Pineapple yellow
    # '#F08A5D',  # Papaya
    # '#EB6A80',  # Watermelon pink
    # '#C875C8',  # Light dragonfruit (softer, lavender-tinted)
    # '#8D75C9',  # Grape soda (cool, but not shadowy)
    # '#7088B8',  # Breezy navy (lighter and more serene)
    # '#3FEEE6',  # Mint splash
    # '#66D3E9',  # Sky aqua
    # '#A9D8F7',  # Tropical blue
    # '#B9F6CA'   # Mint leaf (more vibrant than Coconut mist)


    
    '#F9ED69',  # Amazon
    '#F08A5D',  # Anthropic
    '#EB6A80',  # Cohere
    '#C875C8',  # Deepseek
    '#8D75C9',  # Google
    '#7088B8',  # Meta
    '#3FEEE6',  # Mistral
    '#66D3E9',  # OpenAI
    '#A9D8F7',  # Qwen
    '#B9F6CA'   # XAI


    # 'OPENAI': '#10A37F',       # OpenAI green
    # 'ANTHROPIC': '#9933CC',    # Anthropic purple
    # 'GOOGLE': '#4285F4',       # Google blue
    # 'MISTRAL': '#00A4FF',      # Mistral blue
    # 'DEEPSEEK': '#FF5722',     # DeepSeek orange
    # 'COHERE': '#8A2BE2',       # Cohere purple
    # 'XAI': '#B22222',          # xAI blue
    # 'META': '#0668E1',         # Meta blue
    # 'OPENROUTER': '#FF6B6B',   # OpenRouter red
    # 'QWEN': '#00CDAC',        
    
    
    # '#F9ED69',  # Amazon
    # '#9933CC',  # Anthropic
    # '#8A2BE2',  # Cohere
    # '#FF5722',  # Deepseek
    # '#4285F4',  # Google
    # '#0668E1',  # Meta
    # '#00A4FF',  # Mistral
    # '#10A37F',  # OpenAI
    # '#00CDAC',  # Qwen
    # '#B22222'   # XAI




    # Alternative options for the last color:
    # '#80DEEA'   # Aqua blue (more saturated turquoise)
    # '#C5E1A5'   # Light lime (fresh, vibrant green)
    # '#81C784'   # Jade green (more muted but still visible)
    # '#90CAF9'   # Sky blue (matches the blue theme)
    # '#FFE082'   # Light amber (warm contrast to the blues)
    # '#B39DDB'   # Light purple (complements the grape colors)


    # '#FDF5E6',  # Coconut cream
    # '#FFD9C0',  # Papaya milk
    # '#F7A1A1',  # Strawberry glaze
    # '#E28D9B',  # Rosé
    # '#CD94E7',  # Violet sugar
    # '#B4B4F9',  # Lavender mist
    # '#9DD6F9',  # Blue punch
    # '#90F1EF',  # Mint jelly
    # '#C1FFD7',  # Lime foam
    # '#FFFBDA'   # Banana mochi
    
    # '#FCE38A',  # Lemon sorbet
    # '#F38181',  # Strawberry ice
    # '#EA9AB2',  # Berry mousse
    # '#B083EF',  # Lavender dream
    # '#A0E7E5',  # Sea glass
    # '#8EC6FF',  # Sky jelly
    # '#A7FFD8',  # Lime gelato
    # '#FCD5CE',  # Rosewater
    # '#FFF6E5',  # Peach sherbet
    # '#D5AAFF'   # Taro swirl




    # '#FFEDB4',  # Lemon zest
    # '#FFC3A1',  # Papaya milk
    # '#F5A3B7',  # Strawberry ice
    # '#E59BE9',  # Dragonfruit gelato
    # '#B28DFF',  # Grape breeze
    # '#A3C9F9',  # Sky blue
    # '#90E0EF',  # Seafoam
    # '#CAF7E3',  # Frozen mint
    # '#FAF3DD',  # Pineapple cream
    # '#FFF0F3'   # Cotton cloud



    # '#EAD7C2',  # Sandstone
    # '#DDBEA9',  # Clay
    # '#C89F9C',  # Dusty rose
    # '#A26769',  # Terracotta
    # '#582C4D',  # Mulberry
    # '#4F3824',  # Coffee bark
    # '#7A6C5D',  # Taupe
    # '#A2A392',  # Sage
    # '#DDE0BD',  # Olive mist
    # '#F8F1E5'   # Vanilla cream
    
    # '#E0F7FA',  # Frost
    # '#B2EBF2',  # Polar ice
    # '#81D4FA',  # Glacial blue
    # '#4FC3F7',  # Arctic sky
    # '#29B6F6',  # Crisp blue
    # '#03A9F4',  # Blue glacier
    # '#0288D1',  # Deep water
    # '#0277BD',  # Ice cave
    # '#01579B',  # Ocean trench
    # '#003C8F'   # Glacial abyss
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
    ax.set_title('ResearchCodeBench', fontweight='bold')
    
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
