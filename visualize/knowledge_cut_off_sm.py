import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import json
import sys
import re
from datetime import datetime
import os
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

# Set font and style for publication quality
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

# Add better colors for readability
DEVELOPER_COLORS = {
    'OPENAI': '#10A37F',       # OpenAI green
    'ANTHROPIC': '#9933CC',    # Anthropic purple
    'GOOGLE': '#4285F4',       # Google blue
    'MISTRAL': '#00A4FF',      # Mistral blue
    'DEEPSEEK': '#FF5722',     # DeepSeek orange
    'COHERE': '#8A2BE2',       # Cohere purple
    'XAI': '#36454F',          # xAI blue
    'META': '#0668E1',         # Meta blue
    'OPENROUTER': '#FF6B6B',   # OpenRouter red
    'QWEN': '#00CDAC',         # Qwen teal
}

# Color palette for years with more distinct colors
YEAR_COLORS = {
    2020: "#E6F7FF",  # Light blue
    2021: "#E6FFE6",  # Light green
    2022: "#FFF8E6",  # Light amber/yellow
    2023: "#FFE6E6",  # Light red/pink
    2024: "#E6F7FF",  # Light purple
    2025: "#FFF8E6",  # Light magenta
}

# Path to the overall_stats.json file (will be searched for if not found)
OVERALL_STATS_PATH = 'outputs/20llms_greedy/2025-05-12-17-13-20/overall_stats.json'

def parse_date(date_obj):
    """Parse date objects in various formats."""
    try:
        if isinstance(date_obj, str):
            # Manually parse the date string
            if len(date_obj.split('-')) == 2:
                # Handle YYYY-MM format
                return datetime.strptime(date_obj + "-01", "%Y-%m-%d")
            else:
                # Handle YYYY-MM-DD format
                return datetime.strptime(date_obj, "%Y-%m-%d")
        elif isinstance(date_obj, datetime):
            # If it's already a datetime object, return it directly
            return date_obj
        elif hasattr(date_obj, 'year') and hasattr(date_obj, 'month') and hasattr(date_obj, 'day'):
            # If it's a date object, convert to datetime
            return datetime(date_obj.year, date_obj.month, date_obj.day)
        return pd.NaT  # Return a pandas NaT (Not a Time) for invalid dates
    except (ValueError, TypeError) as e:
        print(f"Error parsing date '{date_obj}': {e}")
        return pd.NaT

def find_overall_stats_file():
    """Find the overall_stats.json file by searching in common locations."""
    # Check if the default path exists
    if os.path.exists(OVERALL_STATS_PATH):
        return OVERALL_STATS_PATH
    
    # Look in all subdirectories of outputs/20llms_greedy
    base_dir = 'outputs/20llms_greedy'
    if os.path.exists(base_dir):
        for subdir in sorted(os.listdir(base_dir), reverse=True):  # Sort to get newest first
            potential_path = os.path.join(base_dir, subdir, 'overall_stats.json')
            if os.path.exists(potential_path):
                print(f"Found overall_stats.json at: {potential_path}")
                return potential_path
    
    # If not found in default locations, look in the entire outputs directory
    for root, dirs, files in os.walk('outputs'):
        for file in files:
            if file == 'overall_stats.json':
                path = os.path.join(root, file)
                print(f"Found overall_stats.json at: {path}")
                return path
    
    print("Warning: Could not find overall_stats.json file.")
    return None

def load_llm_data_from_overall_stats():
    """Load LLM data from the overall_stats.json file."""
    stats_file = find_overall_stats_file()
    if not stats_file:
        print("Error: Could not find overall_stats.json. Will use default knowledge cutoff dates.")
        return pd.DataFrame()
    
    print(f"Loading model information from {stats_file}")
    with open(stats_file, 'r') as f:
        data = json.load(f)
    
    llm_data = []
    for model_name, model_info in data.get('overall_scores', {}).items():
        if 'llm_cfg' in model_info:
            llm_cfg = model_info['llm_cfg']
            
            # Extract model information from llm_cfg
            knowledge_cutoff = llm_cfg.get('knowledge_cutoff_date')
            developer = llm_cfg.get('developer', '').upper()
            print(developer)
            # breakpoint()
            pretty_name = llm_cfg.get('pretty_name', model_name.replace('_', ' ').title())
            
            if not developer:
                developer = llm_cfg.get('company', '').upper()
            
            if not developer:
                developer = model_name.split('_')[0] if '_' in model_name else model_name
            
            # Skip Mistral AI models as their knowledge cutoff is unclear
            if developer == "MISTRAL" or "MISTRAL" in model_name.upper():
                print(f"Skipping Mistral AI model: {model_name} (knowledge cutoff unclear)")
                continue
                
            # Skip Amazon models as their knowledge cutoff is unclear
            if developer == "AMAZON" or "COHERE" in model_name.upper():
                print(f"Skipping Amazon and Cohere model: {model_name} (knowledge cutoff unclear)")
                continue
            
            if knowledge_cutoff:
                cutoff_date = parse_date(knowledge_cutoff)
                if not pd.isna(cutoff_date):
                    llm_data.append({
                        'llm': model_name,
                        'cutoff_date': cutoff_date,
                        'developer': developer,
                        'pretty_name': pretty_name
                    })
                    
    return pd.DataFrame(llm_data)

def extract_model_info_from_name(model_name):
    """Extract model developer from its name."""
    if "CLAUDE" in model_name:
        return "ANTHROPIC"
    elif "GPT" in model_name or "O1" in model_name or "O3" in model_name or "O4" in model_name:
        return "OPENAI"
    elif "GEMINI" in model_name:
        return "GOOGLE"
    elif "MISTRAL" in model_name:
        return "MISTRAL"
    elif "DEEPSEEK" in model_name:
        return "DEEPSEEK"
    elif "GROK" in model_name:
        return "XAI"
    elif "COHERE" in model_name:
        return "COHERE"
    elif "LLAMA" in model_name:
        return "META"
    elif "QWEN" in model_name:
        return "QWEN"
    else:
        # For other models, extract the first part of the name as the developer
        return model_name.split('_')[0] if '_' in model_name else model_name

def extract_date_from_model_name(model_name):
    """Try to extract a date from the model name."""
    # Look for patterns like YYYY_MM_DD or YYYY-MM-DD
    date_patterns = [
        r'(\d{4})[-_](\d{1,2})[-_](\d{1,2})',  # YYYY-MM-DD or YYYY_MM_DD
        r'(\d{4})(\d{2})(\d{2})',              # YYYYMMDD
        r'(\d{4})[-_](\d{1,2})'                # YYYY-MM or YYYY_MM
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, model_name)
        if match:
            groups = match.groups()
            try:
                if len(groups) == 3:  # YYYY-MM-DD format
                    year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                    return f"{year}-{month:02d}-{day:02d}"
                elif len(groups) == 2:  # YYYY-MM format
                    year, month = int(groups[0]), int(groups[1])
                    return f"{year}-{month:02d}"
            except ValueError:
                pass
    return None

def get_model_knowledge_cutoff(model_name):
    """Determine knowledge cutoff date for a model based on its name and known patterns."""
    # Default knowledge cutoff dates based on model families
    # if "CLAUDE_3_5" in model_name or "CLAUDE_3_7" in model_name:
    #     return "2023-08"
    # elif "GPT_4O" in model_name:
    #     return "2023-10"
    # elif "GPT_4_1" in model_name:
    #     return "2023-12"
    # elif "O1" in model_name or "O3" in model_name or "O4" in model_name:
    #     return "2023-10"
    # elif "GEMINI_2_0" in model_name:
    #     return "2024-06"
    # elif "GEMINI_2_5" in model_name:
    #     return "2024-08"
    # elif "GROK_3" in model_name:
    #     return "2024-11"
    # elif "DEEPSEEK" in model_name:
    #     return "2023-11"
    # elif "MISTRAL" in model_name or "CODESTRAL" in model_name:
    #     return "2023-10"
    # elif "LLAMA_3" in model_name:
    #     return "2023-09"
    # elif "LLAMA_4" in model_name:
    #     return "2024-02"
    # elif "QWEN" in model_name:
    #     return "2023-10"
    
    # Try to extract date from model name as fallback
    extracted_date = extract_date_from_model_name(model_name)
    if extracted_date:
        return extracted_date
    
    # If no date could be determined, use a default
    return "2023-10"  # Default fallback

def load_llm_data_from_json(json_file):
    """Extract LLM data directly from the JSON file."""
    # First try to load from overall_stats.json
    llm_df = load_llm_data_from_overall_stats()
    # print(llm_df)
    # breakpoint()
    # If we got data from overall_stats.json, use it
    if not llm_df.empty:
        return llm_df
    
    # Otherwise fall back to extracting from the provided JSON file
    print("Falling back to extracting model info from the provided JSON file.")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Get all unique model names from the JSON file
    model_names = set()
    for paper_info in data['results'].values():
        for model_name in paper_info.get('results', {}).keys():
            model_names.add(model_name)
    
    # Create LLM data with knowledge cutoff dates
    llm_data = []
    for model_name in model_names:
        developer = extract_model_info_from_name(model_name)
        breakpoint()
        
        # Skip Mistral AI models as their knowledge cutoff is unclear
        if developer == "MISTRAL" or "MISTRAL" in model_name.upper():
            print(f"Skipping Mistral AI model: {model_name} (knowledge cutoff unclear)")
            continue
            
        # Skip Amazon models as their knowledge cutoff is unclear
        if developer == "AMAZON" or "CLAUDE" in model_name.upper() or "CLAUDE" in developer:
            print(f"Skipping Amazon model: {model_name} (knowledge cutoff unclear)")
            continue
            
        knowledge_cutoff = get_model_knowledge_cutoff(model_name)
        
        cutoff_date = parse_date(knowledge_cutoff)
        if cutoff_date:
            llm_data.append({
                'llm': model_name,
                'cutoff_date': cutoff_date,
                'developer': developer,
                'pretty_name': model_name.replace('_', ' ').title()
            })
    
    return pd.DataFrame(llm_data)

def load_papers_data(json_file):
    """Load paper data from the JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    papers = []
    for paper_id, paper_info in data['results'].items():
        if 'paper_metadata' in paper_info:
            metadata = paper_info['paper_metadata']
            papers.append({
                'id': paper_id,
                'title': metadata.get('title', 'Unknown'),
                'first_commit_date': parse_date(metadata.get('first_commit_date')),
                'last_commit_date': parse_date(metadata.get('last_commit_date')),
                'venue': metadata.get('venue', 'Unknown')
            })
    
    return pd.DataFrame(papers)

def create_horizontal_visualization(papers_df, llm_df, output_path):
    """Create a horizontal visualization with papers on x-axis and timeline on y-axis."""
    # Sort papers by first commit date
    papers_df = papers_df.sort_values(by='first_commit_date')
    
    # Drop papers without commit dates
    papers_df = papers_df.dropna(subset=['first_commit_date', 'last_commit_date'])
    
    # Create a figure with appropriate dimensions for horizontal layout
    fig = plt.figure(figsize=(7, 4), dpi=300, constrained_layout=True)
    
    # Create a simple layout with just the main plot
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0, 0])  # Main plot takes the entire figure
    
    # Get all dates for setting the y-axis limits
    all_dates = []
    all_dates.extend(llm_df['cutoff_date'].dropna())
    all_dates.extend(papers_df['first_commit_date'].dropna())
    all_dates.extend(papers_df['last_commit_date'].dropna())
    
    # Set y-axis limits with some padding
    min_date = min(all_dates) + 1* pd.DateOffset(months=1)
    # breakpoint()
    max_date = max(all_dates) + pd.DateOffset(months=1)
    
    # Create a clean white background
    ax.set_facecolor('white')
    
    # Add date grid lines
    month_intervals = 2
    date_range = pd.date_range(start=min_date, end=max_date, freq=f'{month_intervals}MS')
    for date in date_range:
        ax.axhline(date, color='#eeeeee', linestyle='-', linewidth=0.5, zorder=-2)
    
    # Add colored year backgrounds and year boundary lines
    year_boundaries = {}
    min_year = min(d.year for d in all_dates if not pd.isna(d))
    max_year = max(d.year for d in all_dates if not pd.isna(d))
    
    # Add a colored background for each year
    for year in range(min_year, max_year + 1):
        year_start = pd.Timestamp(f"{year}-01-01")
        year_end = pd.Timestamp(f"{year + 1}-01-01") if year < max_year else max_date
        
        if year_start <= max_date and year_end >= min_date:
            # Get color for this year (default to a light gray if not in the color dict)
            color = YEAR_COLORS.get(year, "#f5f5f5")
            
            # Create a rectangle spanning the full width of the plot for this year
            # Convert timestamps to matplotlib date numbers for Rectangle
            year_start_num = mdates.date2num(year_start)
            year_end_num = mdates.date2num(year_end)
            height = year_end_num - year_start_num
            
            rect = Rectangle((0, year_start_num), len(papers_df) + 1, height,
                           facecolor=color, alpha=0.5, zorder=-3)  # Increased alpha for more saturation
            ax.add_patch(rect)
            
            # Add year boundary line (darker)
            if year_start >= min_date:
                ax.axhline(year_start, color='#777777', linestyle='-', linewidth=1.2, zorder=-1)
                
                # Add year label in the MIDDLE with a smaller, more subtle design
                ax.text(0.5, year_start + pd.DateOffset(days=15), str(year), fontsize=9, 
                       ha='center', va='center', color='#222222', fontweight='bold',
                       transform=ax.get_yaxis_transform(),
                       bbox=dict(facecolor='white', alpha=0.6, edgecolor=None, pad=1))
                year_boundaries[year] = True
    
    # Ensure 2023 is labeled in the middle-bottom if it's in the range
    if 2023 >= min_year and 2023 <= max_year:
        year_2023_start = pd.Timestamp("2023-01-01")
        if year_2023_start < min_date:
            # If 2023-01-01 is before our min_date, use the visible part of 2023
            year_2023_position = min_date + pd.DateOffset(days=15)
        else:
            # Position the label near the end of 2023
            year_2023_position = pd.Timestamp("2023-12-15")
            
        # Only add if the position is within our visible range
        if year_2023_position >= min_date and year_2023_position <= max_date:
            ax.text(0.5, year_2023_position, "2023", fontsize=9,
                   ha='center', va='center', color='#222222', fontweight='bold',
                   transform=ax.get_yaxis_transform(),
                   bbox=dict(facecolor='white', alpha=0.6, edgecolor=None, pad=1))
    
    # Plot LLM knowledge cutoff dates as horizontal lines
    developer_handles = {}
    for developer, group in llm_df.groupby('developer'):
        # print(developer)
        color = DEVELOPER_COLORS.get(developer, '#999999')
        dates = sorted(group['cutoff_date'].dropna())
        
        # Only draw one line per developer with the most recent date
        if dates:
            latest_date = max(dates)
            line = ax.axhline(latest_date, color=color, linestyle='-', alpha=0.6, linewidth=1.2)
            developer_handles[developer] = line
            
            # Add developer name to the RIGHT side of the plot instead of left
            if developer == "XAI":
                # Draw XAI on the left side with custom coordinates
                left_position = -0.037  # Adjust this value to move XAI text left/right
                ax.text(left_position, latest_date - pd.DateOffset(days=3), developer, color=color, 
                       ha='left', va='top', fontweight='bold', fontsize=9,
                       transform=ax.get_yaxis_transform())
            else:
                # Draw all other developers on the right side
                ax.text(0.99, latest_date - pd.DateOffset(days=3), developer, color=color, 
                       ha='right', va='top', fontweight='bold', fontsize=9,
                       transform=ax.get_yaxis_transform())
    
    # Add papers as vertical timelines
    for i, (_, paper) in enumerate(papers_df.iterrows()):
        x_pos = i + 1  # Position each paper on the x-axis
        
        # Count how many LLMs have knowledge cutoffs before paper dates
        cutoffs_before_first_commit = sum(llm_df['cutoff_date'] <= paper['first_commit_date']) if not pd.isna(paper['first_commit_date']) else 0
        cutoffs_before_last_commit = sum(llm_df['cutoff_date'] <= paper['last_commit_date']) if not pd.isna(paper['last_commit_date']) else 0
        
        pct_before_first_commit = 100 * cutoffs_before_first_commit / len(llm_df) if not pd.isna(paper['first_commit_date']) else 0
        pct_before_last_commit = 100 * cutoffs_before_last_commit / len(llm_df) if not pd.isna(paper['last_commit_date']) else 0
        
        # Draw line for the timeline from first to last commit
        if not pd.isna(paper['first_commit_date']) and not pd.isna(paper['last_commit_date']):
            ax.plot([x_pos, x_pos], [paper['first_commit_date'], paper['last_commit_date']], 
                    color='#666666', linewidth=1.0, alpha=0.7)
        # Mark last commit date with star
        if not pd.isna(paper['last_commit_date']):
            ax.scatter(x_pos, paper['last_commit_date'], color='#4CAF50', s=80, marker='*', zorder=5, alpha=0.8)
    
        # Mark first commit date with square
        if not pd.isna(paper['first_commit_date']):
            ax.scatter(x_pos, paper['first_commit_date'], color='#DB4437', s=60, marker='*', zorder=5, alpha=0.8)
        

    # Create legend elements
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#DB4437', markersize=14, label='First Code Commit'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#4CAF50', markersize=14, label='Last Code Commit'),
        # Line2D([0], [0], marker='>', color='w', markerfacecolor='gray', markersize=8, label='Knowledge Cutoff of Models')
    ]
    
    # Simple legend placement with figure coordinates (easy to adjust manually)
    ax.legend(handles=legend_elements, 
              loc='lower left',            # Basic position anchor
              bbox_to_anchor=(0.755, 0.155),   # Simple x,y coordinates in figure fraction (0-1 range)
              frameon=True, 
              framealpha=0.8, 
              ncol=1,
              fontsize=10,
              # Control box size with these parameters:
              borderpad=0.5,        # Padding between edge of box and legend contents (smaller=tighter)
              handlelength=1.0,     # Length of legend handles (smaller=narrower box)
              labelspacing=0.2,     # Vertical spacing between legend entries (smaller=shorter box)
              handletextpad=0.4)    # Space between handle and text (smaller=narrower box)
    
    # Set x-ticks at paper positions and label them with paper IDs
    ax.set_xticks(range(1, len(papers_df) + 1))
    
    # Create shorter paper ID labels (truncate long IDs)
    id2pretty_name = {
    "advantage-alignment": "Duque et al.",
    "Diff-Transformer": "Ye et al.",
    "DiffusionDPO": "Wallace et al.",
    "DyT": "Zhu et al.",
    "eomt": "Kerssies et al.",
    "fractalgen": "Li et al.",
    "GMFlow": "Chen et al.",
    "GPS": "Zhang et al.",
    "grid-cell-conformal-isometry": "Xu et al.",
    "hyla": "Schug et al.",
    "LEN": "Chen et al.",
    "llm-sci-use": "Liang et al.",
    "minp": "Nguyen et al.",
    "OptimalSteps": "Pei et al.",
    "REPA-E": "Leng et al.",
    "schedule_free": "Defazio et al.",
    "semanticist": "Wen et al.",
    "SISS": "Alberti et al.",
    "TabDiff": "Shi et al.",
    "Tanh-Init": "Lee et al."
    }
    paper_labels = [f"{id2pretty_name[paper['id']]}" if paper['id'] in id2pretty_name else paper['id'] for _, paper in papers_df.iterrows()]

    ax.set_xticklabels(paper_labels, rotation=45, ha='right', fontsize=9)
    
    # Remove x and y-axis ticks, but keep the labels
    ax.tick_params(axis='x', which='both', length=0)
    ax.tick_params(axis='y', which='both', length=0)
    
    # Format the y-axis to show dates properly
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.yaxis.set_major_locator(mdates.MonthLocator(interval=month_intervals))
    
    # Set axis limits
    ax.set_ylim(min_date, max_date)
    ax.set_xlim(0, len(papers_df) + 1)
    
    # Adjust layout and save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    # Also save as PDF for publication quality
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"Visualization also saved as {pdf_path}")
    
    plt.close()
    
    # Print a summary of the analysis
    print("\nSummary Analysis:")
    print(f"Total number of models with knowledge cutoff dates: {len(llm_df)}")
    print(f"Total number of papers analyzed: {len(papers_df)}")
    
    # Print the unique knowledge cutoff dates
    print("\nUnique knowledge cutoff dates:")
    for date in sorted(llm_df['cutoff_date'].unique()):
        count = sum(llm_df['cutoff_date'] == date)
        print(f"  {date.strftime('%Y-%m')}: {count} models")
    
    # Calculate average percentage of models with knowledge before commits
    avg_pct_before_first_commit = papers_df.apply(
        lambda paper: 100 * sum(llm_df['cutoff_date'] <= paper['first_commit_date']) / len(llm_df) 
        if not pd.isna(paper['first_commit_date']) else np.nan, axis=1).mean()
    
    avg_pct_before_last_commit = papers_df.apply(
        lambda paper: 100 * sum(llm_df['cutoff_date'] <= paper['last_commit_date']) / len(llm_df) 
        if not pd.isna(paper['last_commit_date']) else np.nan, axis=1).mean()
    
    # Print the percentages
    print(f"Average % of models with knowledge before first commit: {avg_pct_before_first_commit:.1f}%")
    print(f"Average % of models with knowledge before last commit: {avg_pct_before_last_commit:.1f}%")

def main():
    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python knowledge_cut_off_sm.py <path_to_json_file> [output_path]")
        sys.exit(1)
    
    json_file = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "model_commits_timeline.png"
    
    # Load data directly from the JSON file
    llm_df = load_llm_data_from_json(json_file)
    papers_df = load_papers_data(json_file)
    print(llm_df)
    print(papers_df)
    # Create horizontal visualization
    create_horizontal_visualization(papers_df, llm_df, output_path)

if __name__ == "__main__":
    main()
