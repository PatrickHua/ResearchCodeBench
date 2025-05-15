import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd

# Use serif font for publication-ready look
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

# Load the data files
wo_paper_json = json.load(open("outputs/20llms_greedy_wo_paper/2025-05-13-11-43-56/overall_stats.json"))
with_paper_json = json.load(open("outputs/20llms_greedy/2025-05-13-10-05-10/overall_stats.json"))

def extract_llm_data(with_paper_json, wo_paper_json):
    """Extract LLM performance data from both datasets and calculate differences."""
    # Create dictionaries to store the data
    with_paper_data = {}
    wo_paper_data = {}
    
    # Extract data from with paper JSON
    for key, value in with_paper_json['overall_scores'].items():
        with_paper_data[key] = {
            'score': value['line_rates']['mean'],
            'pretty_name': value['llm_cfg']['pretty_name']
        }
    
    # Extract data from without paper JSON
    for key, value in wo_paper_json['overall_scores'].items():
        wo_paper_data[key] = {
            'score': value['line_rates']['mean'],
            'pretty_name': value['llm_cfg']['pretty_name']
        }
    
    # Create lists for data
    llm_ids = []
    with_paper_scores = []
    wo_paper_scores = []
    differences = []
    pretty_names = []
    
    # Find common LLMs and calculate differences
    for key in with_paper_data:
        if key in wo_paper_data:
            llm_ids.append(key)
            with_paper_score = with_paper_data[key]['score']
            wo_paper_score = wo_paper_data[key]['score']
            with_paper_scores.append(with_paper_score)
            wo_paper_scores.append(wo_paper_score)
            differences.append(with_paper_score - wo_paper_score)
            pretty_names.append(with_paper_data[key]['pretty_name'])
    
    # Create a DataFrame for easier manipulation
    df = pd.DataFrame({
        'llm_id': llm_ids,
        'pretty_name': pretty_names,
        'with_paper': with_paper_scores,
        'wo_paper': wo_paper_scores,
        'difference': differences
    })
    
    return df

def plot_llm_performance_difference(df):
    """Create a horizontal bar chart of LLM performance differences."""
    # Sort by difference
    df_sorted = df.sort_values(by='difference', ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create colormap for bars
    colors = plt.cm.coolwarm(np.interp(df_sorted['difference'], [-10, 10], [0, 1]))
    
    # Create horizontal bar chart
    bars = ax.barh(df_sorted['llm_id'], df_sorted['difference'], color=colors)
    
    # Add a vertical line at x=0
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel('Performance Difference with Paper vs. Without Paper (Percentage Points)')
    ax.set_title('LLM Performance Difference with vs. without Paper Context')
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='x')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('llm_paper_impact.png', dpi=300)
    plt.savefig('llm_paper_impact.pdf', bbox_inches='tight')
    
    # Show statistics
    print("\nLLMs most improved with paper context:")
    print(df_sorted.head(10)[['llm_id', 'with_paper', 'wo_paper', 'difference']].to_string(index=False))
    
    print("\nLLMs least improved with paper context:")
    print(df_sorted.tail(10)[['llm_id', 'with_paper', 'wo_paper', 'difference']].to_string(index=False))
    
    return fig

def main():
    print("Analyzing LLM performance differences with and without paper context")
    
    # Extract data and calculate differences
    df = extract_llm_data(with_paper_json, wo_paper_json)
    
    # Create visualization
    fig = plot_llm_performance_difference(df)
    
    # Display overall statistics
    mean_diff = df['difference'].mean()
    median_diff = df['difference'].median()
    positive_count = len(df[df['difference'] > 0])
    negative_count = len(df[df['difference'] < 0])
    neutral_count = len(df[df['difference'] == 0])
    
    print("\nOverall Statistics:")
    print(f"Average performance difference: {mean_diff:.2f} percentage points")
    print(f"Median performance difference: {median_diff:.2f} percentage points")
    print(f"LLMs that perform better with paper: {positive_count}")
    print(f"LLMs that perform better without paper: {negative_count}")
    print(f"LLMs with no difference: {neutral_count}")
    
    # Create a scatter plot comparing with vs without paper performance
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot points
    ax.scatter(df['wo_paper'], df['with_paper'], alpha=0.7, s=60)
    
    # Add text labels for each point
    for i, row in df.iterrows():
        ax.text(row['wo_paper'], row['with_paper'], row['llm_id'], fontsize=8)
    
    # Plot diagonal line (no difference)
    min_val = min(df['with_paper'].min(), df['wo_paper'].min()) - 5
    max_val = max(df['with_paper'].max(), df['wo_paper'].max()) + 5
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    # Set labels and title
    ax.set_xlabel('Success Rate Without Paper Context (%)')
    ax.set_ylabel('Success Rate With Paper Context (%)')
    ax.set_title('LLM Performance Comparison: With vs. Without Paper Context')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('llm_performance_scatter.png', dpi=300)
    plt.savefig('llm_performance_scatter.pdf', bbox_inches='tight')
    
    print("\nVisualization files saved:")
    print("- llm_paper_impact.png/pdf: Bar chart of performance differences")
    print("- llm_performance_scatter.png/pdf: Scatter plot comparing with vs without paper performance")

if __name__ == "__main__":
    main()
