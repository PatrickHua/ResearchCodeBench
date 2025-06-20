import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yaml
import sys
from datetime import datetime
import os
from pathlib import Path
from matplotlib.lines import Line2D

# Add the parent directory to the path so we can import from core
sys.path.append(str(Path(__file__).parent.parent))
from core.data_classes.llm_type import MODEL_CONFIGS, LLMType

# Set font and style for publication quality
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14
})

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

def debug_date_parsing(paper):
    """Debug function to help understand date parsing issues."""
    print("\nDebugging date parsing for paper:", paper.get('id'))
    
    for date_field in ['arxiv_date', 'arxiv_v1_date', 'first_commit_date', 'last_commit_date']:
        date_str = paper.get(date_field)
        print(f"  {date_field}: {date_str} (type: {type(date_str)})")
        
        # Try parsing manually
        if isinstance(date_str, str):
            try:
                parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
                print(f"    Successfully parsed as: {parsed_date}")
            except Exception as e:
                print(f"    Failed to parse: {e}")
        else:
            print(f"    Not a string, skipping manual parse")

def load_papers_data(yaml_path):
    """Load papers data from YAML file."""
    with open(yaml_path, 'r') as file:
        papers = yaml.safe_load(file)
    
    papers_data = []
    for paper in papers:
        # Debug the paper data
        debug_date_parsing(paper)
        
        # Parse the dates
        arxiv_date_str = paper.get('arxiv_date')
        arxiv_v1_date_str = paper.get('arxiv_v1_date')
        first_commit_date_str = paper.get('first_commit_date')
        last_commit_date_str = paper.get('last_commit_date')
        
        print(f"Processing paper {paper.get('id')}: arxiv_date={arxiv_date_str}, arxiv_v1_date={arxiv_v1_date_str}, first_commit_date={first_commit_date_str}")
        
        arxiv_date = parse_date(arxiv_date_str)
        arxiv_v1_date = parse_date(arxiv_v1_date_str)
        first_commit_date = parse_date(first_commit_date_str)
        last_commit_date = parse_date(last_commit_date_str)
        
        print(f"  Parsed dates: arxiv_date={arxiv_date}, arxiv_v1_date={arxiv_v1_date}, first_commit_date={first_commit_date}, last_commit_date={last_commit_date}")
        
        # Check if we have the required dates for visualization
        if pd.notna(arxiv_date) and pd.notna(first_commit_date):
            papers_data.append({
                'id': paper.get('id'),
                'title': paper.get('title'),
                'arxiv_date': arxiv_date,
                'arxiv_v1_date': arxiv_v1_date,
                'first_commit_date': first_commit_date,
                'last_commit_date': last_commit_date,
                'venue': paper.get('venue')
            })
        else:
            print(f"  Skipping paper {paper.get('id')} due to invalid dates: arxiv_date={arxiv_date}, first_commit_date={first_commit_date}")
    
    return pd.DataFrame(papers_data)

def load_llm_data():
    """Extract knowledge cutoff dates from LLM configurations."""
    llm_data = []
    for llm_type, config in MODEL_CONFIGS.items():
        # Skip models from VLLM
        if config.company == 'VLLM':
            continue
            
        # Skip GEMINI models with 2025 cutoff dates as they're outliers
        if llm_type in [LLMType.GEMINI_2_0_FLASH_LITE_PREVIEW_02_05, LLMType.GEMINI_2_5_FLASH_PREVIEW_04_17]:
            continue
            
        if config.knowledge_cutoff_date:
            # Parse the knowledge cutoff date
            cutoff_date = parse_date(config.knowledge_cutoff_date)
            if cutoff_date:
                llm_data.append({
                    'llm': llm_type.name,
                    'cutoff_date': cutoff_date,
                    'company': config.company
                })
    
    return pd.DataFrame(llm_data)

def create_visualization(papers_df, llm_df, output_path, show_percentages=False, use_last_commit=False):
    """Create a publication-quality visualization.
    
    Args:
        papers_df: DataFrame containing paper information
        llm_df: DataFrame containing LLM cutoff dates
        output_path: Path to save the visualization
        show_percentages: Whether to show percentages of LLMs with cutoffs before papers (default: False)
        use_last_commit: Whether to use last_commit_date instead of first_commit_date (default: False)
    """
    # Sort papers by arxiv date for better visualization
    papers_df = papers_df.sort_values(by='arxiv_date')
    
    # Determine which commit date to use
    commit_date_field = 'last_commit_date' if use_last_commit else 'first_commit_date'
    commit_label = 'Last Code Commit' if use_last_commit else 'First Code Commit'
    
    # Filter out papers with missing commit dates
    papers_df = papers_df.dropna(subset=[commit_date_field])
    
    # Create the main figure and axis
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    
    # Sort LLMs by cutoff date
    llm_df = llm_df.sort_values(by=['company', 'cutoff_date'])
    
    # Define company colors
    company_colors = {
        'OPENAI': '#10A37F',  # OpenAI green
        'ANTHROPIC': '#9933CC',  # Anthropic purple
        'GOOGLE': '#4285F4',  # Google blue
        'DEEPSEEK': '#FF5722',  # DeepSeek orange
        'XAI': '#000000',  # xAI black
    }
    
    # Map companies to color
    llm_df['color'] = llm_df['company'].map(lambda x: company_colors.get(x, '#999999'))
    
    # Get min and max dates for better plotting - filter out NaT values
    all_dates = []
    all_dates.extend(list(papers_df['arxiv_date'].dropna()))
    all_dates.extend(list(papers_df['arxiv_v1_date'].dropna()))
    all_dates.extend(list(papers_df[commit_date_field].dropna()))
    all_dates.extend(list(llm_df['cutoff_date'].dropna()))
    
    min_date = min(all_dates) - pd.DateOffset(months=1)
    max_date = max(all_dates) + pd.DateOffset(months=1)
    
    # Create a clean white background
    ax.set_facecolor('white')
    
    # Add subtle year divider lines
    year_boundaries = {}
    for date in sorted(all_dates):
        if pd.isna(date):
            continue
        year = date.year
        if year not in year_boundaries:
            year_start = pd.Timestamp(f"{year}-01-01")
            if year_start >= min_date and year_start <= max_date:
                ax.axvline(year_start, color='#dddddd', linestyle='-', linewidth=0.8, zorder=-1)
                ax.text(year_start, 0.02, str(year), fontsize=9, ha='center', va='bottom', 
                       color='#777777', transform=ax.get_xaxis_transform())
                year_boundaries[year] = True
    
    # Plot LLM knowledge cutoff dates as vertical dashed lines
    company_handles = {}
    for company, group in llm_df.groupby('company'):
        color = company_colors.get(company, '#999999')
        dates = sorted(group['cutoff_date'].dropna())
        
        # Only draw one line per company with the most recent date
        if dates:
            latest_date = max(dates)
            line = ax.axvline(latest_date, color=color, linestyle='-', alpha=0.6, linewidth=1.2)
            company_handles[company] = line
            
            # Add company name next to the line
            ax.text(latest_date, 0.98, company, color=color, 
                   ha='right', va='top', fontweight='bold', fontsize=9, rotation=90,
                   transform=ax.get_xaxis_transform())
    
    # Plot individual LLM knowledge cutoff dates at the bottom
    # Define the y-position for the LLM markers - move them further down
    llm_y_pos = -0.06  # Further below the x-axis
    
    # Add a thin horizontal line to separate the LLM markers from the main plot
    ax.axhline(y=-0.02, color='#dddddd', linestyle='-', linewidth=0.8, zorder=1)
    
    # Group LLMs by company and unique cutoff dates 
    for company, group in llm_df.groupby('company'):
        color = company_colors.get(company, '#999999')
        
        # Get unique cutoff dates for this company
        unique_dates = group['cutoff_date'].dropna().unique()
        
        # Plot a marker for each unique cutoff date
        for date in unique_dates:
            # Count models with this cutoff date for marker size
            count = sum(group['cutoff_date'] == date)
            size = max(40, 40 + (count * 8))  # Increase base size for better visibility
            
            # Plot the marker - use uniform size and no numbers
            ax.scatter(date, llm_y_pos, color=color, s=70, marker='v', # Fixed size, no scaling
                      alpha=0.8, zorder=5, edgecolors='white', linewidth=0.8)
            
    # Add a label for the LLM markers
    ax.text(min_date, llm_y_pos, "LLM Knowledge\nCutoff Dates:", ha='right', va='center', 
           fontsize=8, fontweight='bold', color='#333333')
    
    # Plot papers as horizontal lines with markers
    for i, paper in enumerate(papers_df.iterrows()):
        idx = i + 1  # offset for better spacing
        paper = paper[1]  # Get the Series
        
        # Count how many LLMs have knowledge cutoffs before paper publication and implementation
        cutoffs_before_arxiv_v1 = sum(llm_df['cutoff_date'] <= paper['arxiv_v1_date']) if not pd.isna(paper['arxiv_v1_date']) else 0
        cutoffs_before_arxiv = sum(llm_df['cutoff_date'] <= paper['arxiv_date'])
        cutoffs_before_commit = sum(llm_df['cutoff_date'] <= paper[commit_date_field])
        
        pct_llms_before_arxiv_v1 = 100 * cutoffs_before_arxiv_v1 / len(llm_df) if not pd.isna(paper['arxiv_v1_date']) else 0
        pct_llms_before_arxiv = 100 * cutoffs_before_arxiv / len(llm_df)
        pct_llms_before_commit = 100 * cutoffs_before_commit / len(llm_df)
        
        # Draw line for the full timeline from v1 (or arxiv date if v1 not available) to commit date
        start_date = paper['arxiv_v1_date'] if not pd.isna(paper['arxiv_v1_date']) else paper['arxiv_date']
        if not pd.isna(start_date) and not pd.isna(paper[commit_date_field]):
            ax.plot([start_date, paper[commit_date_field]], [idx, idx], 
                    color='#aaaaaa', linewidth=0.8, alpha=0.6)
        
        # Mark arxiv v1 date with a diamond if available
        if not pd.isna(paper['arxiv_v1_date']):
            ax.scatter(paper['arxiv_v1_date'], idx, color='#9370DB', s=40, marker='D', zorder=5, alpha=0.8)  # Purple diamond
        
        # Mark latest arxiv date with circle
        if not pd.isna(paper['arxiv_date']):
            ax.scatter(paper['arxiv_date'], idx, color='#00CCCC', s=40, zorder=5, alpha=0.8)  # Teal/cyan
        
        # Mark commit date with star
        if not pd.isna(paper[commit_date_field]):
            ax.scatter(paper[commit_date_field], idx, color='#DB4437', s=60, 
                      marker='*', zorder=5, alpha=0.8)
        
        # Add paper ID on the left
        ax.text(min_date, idx, paper['id'], 
                ha='right', va='center', fontsize=9, 
                color='#333333')
        
        # Only add percentages if show_percentages is True
        if show_percentages:
            if not pd.isna(paper['arxiv_v1_date']):
                percentage_text = f"{pct_llms_before_arxiv_v1:.0f}% | {pct_llms_before_arxiv:.0f}% | {pct_llms_before_commit:.0f}%"
            else:
                percentage_text = f"{pct_llms_before_arxiv:.0f}% | {pct_llms_before_commit:.0f}%"
            ax.text(max_date, idx, percentage_text, 
                    ha='left', va='center', fontsize=8, 
                    color='#555555')
        
        # Add subtle date annotations
        if i % 2 == 0:  # add dates for every other paper to reduce clutter
            if not pd.isna(paper['arxiv_v1_date']):
                ax.text(paper['arxiv_v1_date'], idx + 0.3, f"{paper['arxiv_v1_date'].strftime('%Y-%m')}", 
                       ha='center', va='bottom', fontsize=7, color='#777777')
            if not pd.isna(paper['arxiv_date']):
                ax.text(paper['arxiv_date'], idx + 0.3, f"{paper['arxiv_date'].strftime('%Y-%m')}", 
                       ha='center', va='bottom', fontsize=7, color='#777777')
    
    # Create simple legend with additional explanation for the percentages if needed
    legend_elements = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#9370DB', markersize=7, label='First Arxiv Version'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#00CCCC', markersize=7, label='Latest Arxiv Version'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#DB4437', markersize=12, label=commit_label),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='gray', markersize=7, label='Knowledge Cutoff of Models')
    ]
    
    # Add percentage explanation to legend if showing percentages
    if show_percentages:
        legend_elements.append(
            Line2D([0], [0], marker='', color='w', label='Right values: X% | Y% | Z% = % of LLMs with cutoff before v1 | latest version | implementation')
        )
    
    # Add the legend in the top left corner
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.01, 0.99),
              frameon=True, framealpha=0.95, ncol=1)
    
    # Configure the main axis
    ax.set_ylim(-0.09, len(papers_df) + 1)  # Extend y-axis further below 0 to make room for LLM markers
    ax.set_xlim(min_date, max_date)
    
    # Set the x-axis to show dates nicely
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45, ha='right')
    
    # Hide y-axis ticks and labels
    ax.set_yticks([])
    ax.set_xlabel('Date', fontsize=10, labelpad=25)  # Increase labelpad more for the LLM markers
    
    # Add title with proper spacing
    commit_type = "Last" if use_last_commit else "First"
    ax.set_title(f'Research Paper Timeline vs. LLM Knowledge Cutoff Dates (Using {commit_type} Commit)', pad=20, fontsize=14)
    
    # Adjust layout and save with default parameters
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")

    # Also save as PNG if the output is PDF or vice versa
    if output_path.suffix.lower() == '.pdf':
        png_path = output_path.with_suffix('.png')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"Visualization also saved as {png_path}")
    elif output_path.suffix.lower() == '.png':
        pdf_path = output_path.with_suffix('.pdf')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight') 
        print(f"Visualization also saved as {pdf_path}")
    plt.close()

def calculate_statistics(papers_df, llm_df, use_last_commit=False):
    """Calculate and print statistics about the relationship between papers and LLM cutoff dates."""
    # If no papers were loaded, return empty stats
    if len(papers_df) == 0:
        print("No papers with valid dates were found. Cannot calculate statistics.")
        return pd.DataFrame()
    
    # Determine which commit date to use
    commit_date_field = 'last_commit_date' if use_last_commit else 'first_commit_date'
    commit_type = "last" if use_last_commit else "first"
    
    # Filter papers to only include those with valid dates
    papers_df = papers_df.dropna(subset=['arxiv_date', commit_date_field])
    print(f"Calculating statistics for {len(papers_df)} papers with valid dates")
    
    # For each paper, check how many LLM cutoff dates it comes after
    stats = []
    
    for _, paper in papers_df.iterrows():
        arxiv_date = paper['arxiv_date']
        arxiv_v1_date = paper['arxiv_v1_date'] if not pd.isna(paper['arxiv_v1_date']) else arxiv_date
        commit_date = paper[commit_date_field]
        
        cutoffs_before_arxiv_v1 = sum(llm_df['cutoff_date'] < arxiv_v1_date)
        cutoffs_before_arxiv = sum(llm_df['cutoff_date'] < arxiv_date)
        cutoffs_before_commit = sum(llm_df['cutoff_date'] < commit_date)
        
        stats.append({
            'paper_id': paper['id'],
            'title': paper['title'],
            'venue': paper.get('venue', ''),
            'cutoffs_before_arxiv_v1': cutoffs_before_arxiv_v1,
            'cutoffs_before_arxiv': cutoffs_before_arxiv,
            'cutoffs_before_commit': cutoffs_before_commit,
            'pct_llms_before_v1': round(100 * cutoffs_before_arxiv_v1 / len(llm_df), 1),
            'pct_llms_with_knowledge': round(100 * cutoffs_before_arxiv / len(llm_df), 1),
            'pct_llms_before_implementation': round(100 * cutoffs_before_commit / len(llm_df), 1)
        })
    
    stats_df = pd.DataFrame(stats)
    
    # Calculate overall statistics
    avg_pct_before_arxiv_v1 = stats_df['pct_llms_before_v1'].mean()
    avg_pct_before_arxiv = stats_df['pct_llms_with_knowledge'].mean()
    avg_pct_before_commit = stats_df['pct_llms_before_implementation'].mean()
    
    print(f"\nOn average, {avg_pct_before_arxiv_v1:.1f}% of LLMs have knowledge cutoffs before paper v1 publication")
    print(f"On average, {avg_pct_before_arxiv:.1f}% of LLMs have knowledge cutoffs before latest paper publication")
    print(f"On average, {avg_pct_before_commit:.1f}% of LLMs have knowledge cutoffs before {commit_type} code commit")
    
    # Count papers with implementation after all LLM cutoffs
    papers_after_all_cutoffs = sum(stats_df['cutoffs_before_commit'] == len(llm_df))
    print(f"\n{papers_after_all_cutoffs} papers ({papers_after_all_cutoffs/len(papers_df)*100:.1f}%) "
          f"have {commit_type} commit after all LLM knowledge cutoffs")
    
    # Print details of papers with implementation after all LLM cutoffs
    papers_after_cutoff = stats_df[stats_df['cutoffs_before_commit'] == len(llm_df)]
    if not papers_after_cutoff.empty:
        print("\nPapers with implementation after all LLM knowledge cutoffs:")
        for _, paper in papers_after_cutoff.iterrows():
            print(f"  - {paper['paper_id']}: \"{paper['title']}\" ({paper['venue']})")
    
    return stats_df

def create_sample_data():
    """Create a sample dataset of paper dates for demonstration purposes."""
    sample_papers = [
        {
            'id': 'advantage-alignment',
            'title': 'Advantage Alignment Algorithms',
            'arxiv_date': datetime.strptime('2025-02-06', '%Y-%m-%d'),
            'arxiv_v1_date': datetime.strptime('2024-06-20', '%Y-%m-%d'),
            'first_commit_date': datetime.strptime('2025-04-07', '%Y-%m-%d'),
            'last_commit_date': datetime.strptime('2025-04-07', '%Y-%m-%d'),
            'venue': 'ICLR2025'
        },
        {
            'id': 'Diff-Transformer',
            'title': 'Differential Transformer',
            'arxiv_date': datetime.strptime('2025-04-07', '%Y-%m-%d'),
            'arxiv_v1_date': datetime.strptime('2024-10-07', '%Y-%m-%d'),
            'first_commit_date': datetime.strptime('2024-10-07', '%Y-%m-%d'),
            'last_commit_date': datetime.strptime('2025-03-03', '%Y-%m-%d'),
            'venue': 'ICLR2025'
        },
        {
            'id': 'DiffusionDPO',
            'title': 'Diffusion Model Alignment Using Direct Preference Optimization',
            'arxiv_date': datetime.strptime('2023-11-22', '%Y-%m-%d'),
            'arxiv_v1_date': datetime.strptime('2023-11-22', '%Y-%m-%d'),
            'first_commit_date': datetime.strptime('2023-12-24', '%Y-%m-%d'),
            'last_commit_date': datetime.strptime('2025-02-03', '%Y-%m-%d'),
            'venue': 'arXiv2023'
        },
        {
            'id': 'DyT',
            'title': 'Transformers without Normalization',
            'arxiv_date': datetime.strptime('2025-03-13', '%Y-%m-%d'),
            'arxiv_v1_date': datetime.strptime('2025-03-13', '%Y-%m-%d'),
            'first_commit_date': datetime.strptime('2025-03-09', '%Y-%m-%d'),
            'last_commit_date': datetime.strptime('2025-03-30', '%Y-%m-%d'),
            'venue': 'CVPR2025'
        },
        {
            'id': 'eomt',
            'title': 'Your ViT is Secretly an Image Segmentation Model',
            'arxiv_date': datetime.strptime('2025-03-24', '%Y-%m-%d'),
            'arxiv_v1_date': datetime.strptime('2025-03-24', '%Y-%m-%d'),
            'first_commit_date': datetime.strptime('2025-03-16', '%Y-%m-%d'),
            'last_commit_date': datetime.strptime('2025-04-18', '%Y-%m-%d'),
            'venue': 'CVPR2025'
        },
        {
            'id': 'fractalgen',
            'title': 'Fractal Generative Models',
            'arxiv_date': datetime.strptime('2025-02-25', '%Y-%m-%d'),
            'arxiv_v1_date': datetime.strptime('2025-02-24', '%Y-%m-%d'),
            'first_commit_date': datetime.strptime('2025-02-23', '%Y-%m-%d'),
            'last_commit_date': datetime.strptime('2025-02-25', '%Y-%m-%d'),
            'venue': 'arXiv2025'
        },
        {
            'id': 'GMFlow',
            'title': 'Gaussian Mixture Flow Matching Models',
            'arxiv_date': datetime.strptime('2025-04-07', '%Y-%m-%d'),
            'arxiv_v1_date': datetime.strptime('2025-04-07', '%Y-%m-%d'),
            'first_commit_date': datetime.strptime('2025-04-07', '%Y-%m-%d'),
            'last_commit_date': datetime.strptime('2025-04-24', '%Y-%m-%d'),
            'venue': 'arXiv2025'
        },
        {
            'id': 'GPS',
            'title': 'GPS: A Probabilistic Distributional Similarity',
            'arxiv_date': datetime.strptime('2025-01-22', '%Y-%m-%d'),
            'arxiv_v1_date': datetime.strptime('2025-01-22', '%Y-%m-%d'),
            'first_commit_date': datetime.strptime('2025-02-28', '%Y-%m-%d'),
            'last_commit_date': datetime.strptime('2025-02-28', '%Y-%m-%d'),
            'venue': 'ICLR2025'
        },
        {
            'id': 'grid-cell-conformal-isometry',
            'title': 'On Conformal Isometry of Grid Cells',
            'arxiv_date': datetime.strptime('2025-02-27', '%Y-%m-%d'),
            'arxiv_v1_date': datetime.strptime('2024-05-27', '%Y-%m-%d'),
            'first_commit_date': datetime.strptime('2025-02-18', '%Y-%m-%d'),
            'last_commit_date': datetime.strptime('2025-02-28', '%Y-%m-%d'),
            'venue': 'ICLR2025'
        },
        {
            'id': 'hyla',
            'title': 'Attention as a Hypernetwork',
            'arxiv_date': datetime.strptime('2024-02-17', '%Y-%m-%d'),
            'arxiv_v1_date': datetime.strptime('2024-06-09', '%Y-%m-%d'),
            'first_commit_date': datetime.strptime('2024-06-09', '%Y-%m-%d'),
            'last_commit_date': datetime.strptime('2024-06-22', '%Y-%m-%d'),
            'venue': 'ICLR2025'
        },
        {
            'id': 'LEN',
            'title': 'Second-Order Min-Max Optimization with Lazy Hessians',
            'arxiv_date': datetime.strptime('2025-04-15', '%Y-%m-%d'),
            'arxiv_v1_date': datetime.strptime('2024-10-12', '%Y-%m-%d'),
            'first_commit_date': datetime.strptime('2024-12-01', '%Y-%m-%d'),
            'last_commit_date': datetime.strptime('2025-01-09', '%Y-%m-%d'),
            'venue': 'ICLR2025'
        },
        {
            'id': 'llm-sci-use',
            'title': 'Mapping the increasing use of LLMs in scientific papers',
            'arxiv_date': datetime.strptime('2024-04-01', '%Y-%m-%d'),
            'arxiv_v1_date': datetime.strptime('2024-04-01', '%Y-%m-%d'),
            'first_commit_date': datetime.strptime('2024-05-12', '%Y-%m-%d'),
            'last_commit_date': datetime.strptime('2025-03-20', '%Y-%m-%d'),
            'venue': 'COLM2024'
        },
        {
            'id': 'minp',
            'title': 'Min-p Sampling for Creative and Coherent LLM Outputs',
            'arxiv_date': datetime.strptime('2025-03-20', '%Y-%m-%d'),
            'arxiv_v1_date': datetime.strptime('2024-07-01', '%Y-%m-%d'),
            'first_commit_date': datetime.strptime('2025-03-15', '%Y-%m-%d'),
            'last_commit_date': datetime.strptime('2025-03-15', '%Y-%m-%d'),
            'venue': 'ICLR2025'
        },
        {
            'id': 'OptimalSteps',
            'title': 'Optimal Stepsize for Diffusion Sampling',
            'arxiv_date': datetime.strptime('2025-03-27', '%Y-%m-%d'),
            'arxiv_v1_date': datetime.strptime('2025-03-27', '%Y-%m-%d'),
            'first_commit_date': datetime.strptime('2025-03-27', '%Y-%m-%d'),
            'last_commit_date': datetime.strptime('2025-04-12', '%Y-%m-%d'),
            'venue': 'arXiv2025'
        },
        {
            'id': 'REPA-E',
            'title': 'REPA-E: Unlocking VAE for End-to-End Tuning',
            'arxiv_date': datetime.strptime('2025-04-14', '%Y-%m-%d'),
            'arxiv_v1_date': datetime.strptime('2025-04-14', '%Y-%m-%d'),
            'first_commit_date': datetime.strptime('2025-04-16', '%Y-%m-%d'),
            'last_commit_date': datetime.strptime('2025-04-16', '%Y-%m-%d'),
            'venue': 'arXiv2025'
        },
        {
            'id': 'schedule_free',
            'title': 'The Road Less Scheduled',
            'arxiv_date': datetime.strptime('2024-10-29', '%Y-%m-%d'),
            'arxiv_v1_date': datetime.strptime('2024-05-24', '%Y-%m-%d'),
            'first_commit_date': datetime.strptime('2024-03-31', '%Y-%m-%d'),
            'last_commit_date': datetime.strptime('2025-04-11', '%Y-%m-%d'),
            'venue': 'NeurIPS2024'
        },
        {
            'id': 'semanticist',
            'title': 'Principal Components Enable A New Language of Images',
            'arxiv_date': datetime.strptime('2025-03-11', '%Y-%m-%d'),
            'arxiv_v1_date': datetime.strptime('2025-03-11', '%Y-%m-%d'),
            'first_commit_date': datetime.strptime('2025-03-09', '%Y-%m-%d'),
            'last_commit_date': datetime.strptime('2025-04-17', '%Y-%m-%d'),
            'venue': 'arXiv2025'
        },
        {
            'id': 'SISS',
            'title': 'Data Unlearning in Diffusion Models',
            'arxiv_date': datetime.strptime('2025-03-02', '%Y-%m-%d'),
            'arxiv_v1_date': datetime.strptime('2025-03-02', '%Y-%m-%d'),
            'first_commit_date': datetime.strptime('2025-03-02', '%Y-%m-%d'),
            'last_commit_date': datetime.strptime('2025-03-05', '%Y-%m-%d'),
            'venue': 'ICLR2025'
        },
        {
            'id': 'TabDiff',
            'title': 'TabDiff: a Mixed-type Diffusion Model for Tabular Data Generation',
            'arxiv_date': datetime.strptime('2025-02-16', '%Y-%m-%d'),
            'arxiv_v1_date': datetime.strptime('2024-10-27', '%Y-%m-%d'),
            'first_commit_date': datetime.strptime('2025-02-16', '%Y-%m-%d'),
            'last_commit_date': datetime.strptime('2025-04-12', '%Y-%m-%d'),
            'venue': 'ICLR2025'
        },
        {
            'id': 'Tanh-Init',
            'title': 'Robust Weight Initialization for Tanh Neural Networks',
            'arxiv_date': datetime.strptime('2025-03-02', '%Y-%m-%d'),
            'arxiv_v1_date': datetime.strptime('2024-10-03', '%Y-%m-%d'),
            'first_commit_date': datetime.strptime('2025-02-11', '%Y-%m-%d'),
            'last_commit_date': datetime.strptime('2025-04-21', '%Y-%m-%d'),
            'venue': 'ICLR2025'
        }
    ]
    
    return pd.DataFrame(sample_papers)

def create_commit_visualization(papers_df, llm_df, output_path):
    """Create a visualization focusing only on first and last code commit dates, ignoring arXiv dates.
    
    Args:
        papers_df: DataFrame containing paper information
        llm_df: DataFrame containing LLM cutoff dates
        output_path: Path to save the visualization
    """
    # Filter papers that have both first and last commit dates
    papers_df = papers_df.dropna(subset=['first_commit_date', 'last_commit_date'])
    
    # Sort papers by first_commit_date
    papers_df = papers_df.sort_values(by='first_commit_date')
    
    # Create the main figure and axis
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    
    # Sort LLMs by cutoff date
    llm_df = llm_df.sort_values(by=['company', 'cutoff_date'])
    
    # Define company colors
    company_colors = {
        'OPENAI': '#10A37F',  # OpenAI green
        'ANTHROPIC': '#9933CC',  # Anthropic purple
        'GOOGLE': '#4285F4',  # Google blue
        'DEEPSEEK': '#FF5722',  # DeepSeek orange
        'XAI': '#000000',  # xAI black
    }
    
    # Map companies to color
    llm_df['color'] = llm_df['company'].map(lambda x: company_colors.get(x, '#999999'))
    
    # Get min and max dates for better plotting - filter out NaT values
    all_dates = []
    all_dates.extend(list(papers_df['first_commit_date'].dropna()))
    all_dates.extend(list(papers_df['last_commit_date'].dropna()))
    all_dates.extend(list(llm_df['cutoff_date'].dropna()))
    
    min_date = min(all_dates) - pd.DateOffset(months=1)
    max_date = max(all_dates) + pd.DateOffset(months=1)
    
    # Create a clean white background
    ax.set_facecolor('white')
    
    # Add subtle year divider lines
    year_boundaries = {}
    for date in sorted(all_dates):
        if pd.isna(date):
            continue
        year = date.year
        if year not in year_boundaries:
            year_start = pd.Timestamp(f"{year}-01-01")
            if year_start >= min_date and year_start <= max_date:
                ax.axvline(year_start, color='#dddddd', linestyle='-', linewidth=0.8, zorder=-1)
                ax.text(year_start, 0.02, str(year), fontsize=9, ha='center', va='bottom', 
                       color='#777777', transform=ax.get_xaxis_transform())
                year_boundaries[year] = True
    
    # Plot LLM knowledge cutoff dates as vertical dashed lines
    company_handles = {}
    for company, group in llm_df.groupby('company'):
        color = company_colors.get(company, '#999999')
        dates = sorted(group['cutoff_date'].dropna())
        
        # Only draw one line per company with the most recent date
        if dates:
            latest_date = max(dates)
            line = ax.axvline(latest_date, color=color, linestyle='-', alpha=0.6, linewidth=1.2)
            company_handles[company] = line
            
            # Add company name next to the line
            ax.text(latest_date, 0.98, company, color=color, 
                   ha='right', va='top', fontweight='bold', fontsize=9, rotation=90,
                   transform=ax.get_xaxis_transform())
    
    # Plot individual LLM knowledge cutoff dates at the bottom
    # Define the y-position for the LLM markers - move them further down
    llm_y_pos = -0.06  # Further below the x-axis
    
    # Add a thin horizontal line to separate the LLM markers from the main plot
    ax.axhline(y=-0.02, color='#dddddd', linestyle='-', linewidth=0.8, zorder=1)
    
    # Group LLMs by company and unique cutoff dates 
    for company, group in llm_df.groupby('company'):
        color = company_colors.get(company, '#999999')
        
        # Get unique cutoff dates for this company
        unique_dates = group['cutoff_date'].dropna().unique()
        
        # Plot a marker for each unique cutoff date
        for date in unique_dates:
            # Count models with this cutoff date for marker size
            count = sum(group['cutoff_date'] == date)
            size = max(40, 40 + (count * 8))  # Increase base size for better visibility
            
            # Plot the marker - use uniform size and no numbers
            ax.scatter(date, llm_y_pos, color=color, s=70, marker='v', # Fixed size, no scaling
                      alpha=0.8, zorder=5, edgecolors='white', linewidth=0.8)
            
    # Add a label for the LLM markers
    ax.text(min_date, llm_y_pos, "LLM Knowledge\nCutoff Dates:", ha='right', va='center', 
           fontsize=8, fontweight='bold', color='#333333')
    
    # Plot papers as horizontal lines with markers
    for i, paper in enumerate(papers_df.iterrows()):
        idx = i + 1  # offset for better spacing
        paper = paper[1]  # Get the Series
        
        # Count how many LLMs have knowledge cutoffs before first and last commit
        cutoffs_before_first_commit = sum(llm_df['cutoff_date'] <= paper['first_commit_date'])
        cutoffs_before_last_commit = sum(llm_df['cutoff_date'] <= paper['last_commit_date'])
        
        pct_llms_before_first_commit = 100 * cutoffs_before_first_commit / len(llm_df)
        pct_llms_before_last_commit = 100 * cutoffs_before_last_commit / len(llm_df)
        
        # Draw line from first commit to last commit
        if not pd.isna(paper['first_commit_date']) and not pd.isna(paper['last_commit_date']):
            ax.plot([paper['first_commit_date'], paper['last_commit_date']], [idx, idx], 
                    color='#aaaaaa', linewidth=0.8, alpha=0.6)
        
        # Mark first commit date with star - SWAPPED as requested by user
        if not pd.isna(paper['first_commit_date']):
            ax.scatter(paper['first_commit_date'], idx, color='#DB4437', s=60, marker='*', zorder=5, alpha=0.8)  # Red star
        
        # Mark last commit date with circle - SWAPPED as requested by user
        if not pd.isna(paper['last_commit_date']):
            ax.scatter(paper['last_commit_date'], idx, color='#00CCCC', s=50, marker='o', zorder=5, alpha=0.8)  # Teal circle
        
        # Add paper ID on the left
        ax.text(min_date, idx, paper['id'], 
                ha='right', va='center', fontsize=9, 
                color='#333333')
        
        # Add subtle date annotations
        if i % 2 == 0:  # add dates for every other paper to reduce clutter
            if not pd.isna(paper['first_commit_date']):
                ax.text(paper['first_commit_date'], idx + 0.3, f"{paper['first_commit_date'].strftime('%Y-%m-%d')}", 
                       ha='center', va='bottom', fontsize=7, color='#777777')
            if not pd.isna(paper['last_commit_date']):
                ax.text(paper['last_commit_date'], idx + 0.3, f"{paper['last_commit_date'].strftime('%Y-%m-%d')}", 
                       ha='center', va='bottom', fontsize=7, color='#777777')
    
    # Create legend
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#DB4437', markersize=12, label='First Code Commit'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#00CCCC', markersize=7, label='Last Code Commit'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='gray', markersize=7, label='Knowledge Cutoff of Models')
    ]
    
    # Add the legend in the top left corner
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.01, 0.99),
              frameon=True, framealpha=0.95, ncol=1)
    
    # Configure the main axis
    ax.set_ylim(-0.09, len(papers_df) + 1)  # Extend y-axis further below 0 to make room for LLM markers
    ax.set_xlim(min_date, max_date)
    
    # Set the x-axis to show dates nicely
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45, ha='right')
    
    # Hide y-axis ticks and labels
    ax.set_yticks([])
    ax.set_xlabel('Date', fontsize=10, labelpad=25)  # Increase labelpad more for the LLM markers
    
    # Add title
    ax.set_title('Code Commit Dates vs. LLM Knowledge Cutoff Dates', pad=20, fontsize=14)
    
    # Adjust layout and save with default parameters
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")

    # Also save as PNG if the output is PDF or vice versa
    if output_path.suffix.lower() == '.pdf':
        png_path = output_path.with_suffix('.png')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"Visualization also saved as {png_path}")
    elif output_path.suffix.lower() == '.png':
        pdf_path = output_path.with_suffix('.pdf')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight') 
        print(f"Visualization also saved as {pdf_path}")
    plt.close()

def main():
    """Main function to generate the visualization."""
    # Get file paths
    script_dir = Path(__file__).parent
    output_dir = script_dir / "outputs"
    output_path = output_dir / "knowledge_cutoff_vs_papers.pdf"
    output_path_last_commit = output_dir / "knowledge_cutoff_vs_papers_last_commit.pdf"
    output_path_commits_only = output_dir / "code_commits_vs_llm_cutoffs.pdf"
    yaml_path = script_dir.parent / "pset" / "papers.yaml"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load real data from YAML file
    papers_df = load_papers_data(yaml_path)
    print(f"Loaded {len(papers_df)} papers from {yaml_path}")
    
    # Use sample data as fallback for demonstration if no valid papers were found
    if len(papers_df) == 0:
        papers_df = create_sample_data()
        print(f"Created {len(papers_df)} sample papers for demonstration")
    
    llm_df = load_llm_data()
    print(f"Loaded {len(llm_df)} LLMs with valid knowledge cutoff dates")
    
    # Remove rows with NaT cutoff dates
    llm_df = llm_df.dropna(subset=['cutoff_date'])
    print(f"After removing invalid dates, using {len(llm_df)} LLMs")
    
    # Print some sample LLM data for debugging
    print("\nSample LLM knowledge cutoff dates:")
    for company, group in llm_df.groupby('company'):
        print(f"\n{company} models:")
        for _, row in group.head(3).iterrows():
            if pd.notna(row['cutoff_date']):
                print(f"  - {row['llm']}: {row['cutoff_date'].strftime('%Y-%m-%d')}")
            else:
                print(f"  - {row['llm']}: Invalid date")
    
    # Calculate statistics with first commit date
    print("\n=== Statistics using first commit date ===")
    stats_df = calculate_statistics(papers_df, llm_df, use_last_commit=False)
    
    # Create visualization with first commit date
    create_visualization(papers_df, llm_df, output_path, show_percentages=False, use_last_commit=False)
    
    # Calculate statistics with last commit date
    print("\n=== Statistics using last commit date ===")
    stats_df_last = calculate_statistics(papers_df, llm_df, use_last_commit=True)
    
    # Create visualization with last commit date
    create_visualization(papers_df, llm_df, output_path_last_commit, show_percentages=False, use_last_commit=True)
    
    # Create a visualization focused only on code commit dates
    print("\n=== Creating visualization of code commits only ===")
    create_commit_visualization(papers_df, llm_df, output_path_commits_only)
    
    # Save the statistics
    stats_path = output_dir / "knowledge_cutoff_stats_first_commit.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"Statistics saved to {stats_path}")
    
    stats_path_last = output_dir / "knowledge_cutoff_stats_last_commit.csv"
    stats_df_last.to_csv(stats_path_last, index=False)
    print(f"Statistics saved to {stats_path_last}")

if __name__ == "__main__":
    main()
