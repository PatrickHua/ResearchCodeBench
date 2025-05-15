import matplotlib.pyplot as plt
import json
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

# Set up publication-quality aesthetics
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.linewidth': 1.2,
    'axes.labelsize': 14,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.minor.width': 1.0,
    'ytick.minor.width': 1.0,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'legend.frameon': False,
    'figure.dpi': 300
})

# Load the data files
wo_paper_json = json.load(open("outputs/20llms_greedy_wo_paper/2025-05-13-11-43-56/overall_stats.json"))
with_paper_json = json.load(open("outputs/20llms_greedy/2025-05-13-10-05-10/overall_stats.json"))

# Extract snippets success rates from both datasets
snippets_with_paper = {}
snippets_wo_paper = {}

# Extract data from with paper JSON
for paper_id, paper_data in with_paper_json['results'].items():
    for llm_id, llm_data in paper_data['results'].items():
        for snippet_id, snippet_results in llm_data['results'].items():
            # Create a unique identifier for this snippet
            snippet_key = f"{paper_id}/{snippet_id}"
            
            # Initialize if not already in the dict
            if snippet_key not in snippets_with_paper:
                snippets_with_paper[snippet_key] = {'total': 0, 'passed': 0}
            
            # Add results
            for result in snippet_results:
                snippets_with_paper[snippet_key]['total'] += 1
                if result.get('passed', False):
                    snippets_with_paper[snippet_key]['passed'] += 1

# Extract data from without paper JSON
for paper_id, paper_data in wo_paper_json['results'].items():
    for llm_id, llm_data in paper_data['results'].items():
        for snippet_id, snippet_results in llm_data['results'].items():
            # Create a unique identifier for this snippet
            snippet_key = f"{paper_id}/{snippet_id}"
            
            # Initialize if not already in the dict
            if snippet_key not in snippets_wo_paper:
                snippets_wo_paper[snippet_key] = {'total': 0, 'passed': 0}
            
            # Add results
            for result in snippet_results:
                snippets_wo_paper[snippet_key]['total'] += 1
                if result.get('passed', False):
                    snippets_wo_paper[snippet_key]['passed'] += 1

# Ensure all snippets exist in both datasets
for snippet_key in snippets_with_paper:
    assert snippet_key in snippets_wo_paper

# Calculate success rates for each snippet in both conditions
with_paper_rates = []
wo_paper_rates = []
snippet_ids = []

for snippet_key in snippets_with_paper:
    if snippet_key in snippets_wo_paper:
        # Calculate normalized success rates (percentage of LLMs that passed)
        with_rate = (snippets_with_paper[snippet_key]['passed'] / snippets_with_paper[snippet_key]['total']) * 100
        wo_rate = (snippets_wo_paper[snippet_key]['passed'] / snippets_wo_paper[snippet_key]['total']) * 100
        
        with_paper_rates.append(with_rate)
        wo_paper_rates.append(wo_rate)
        snippet_ids.append(snippet_key)

# Add small jitter to prevent overlap of identical points
jitter = 0.5
wo_paper_rates_jittered = [x + np.random.uniform(-jitter, jitter) for x in wo_paper_rates]
with_paper_rates_jittered = [y + np.random.uniform(-jitter, jitter) for y in with_paper_rates]

# Create color coding based on difference
diff_rates = [w - wo for w, wo in zip(with_paper_rates, wo_paper_rates)]
cmap = plt.cm.RdBu_r
norm = mpl.colors.Normalize(vmin=-20, vmax=20)
colors = [cmap(norm(diff)) for diff in diff_rates]

# Create the plot with improved aesthetics
fig, ax = plt.subplots(figsize=(7, 6))

# Add reference lines
ax.axline((0, 0), slope=1, color='black', linestyle='--', alpha=0.5, zorder=1, linewidth=1.2)
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.2, zorder=1)
ax.axvline(x=0, color='gray', linestyle='-', alpha=0.2, zorder=1)

# Create the scatter plot with color coding
scatter = ax.scatter(wo_paper_rates_jittered, with_paper_rates_jittered, 
                     c=colors, alpha=0.75, s=70, edgecolors='dimgray', 
                     linewidth=0.5, zorder=2)

# Set the axes to be square and with same limits
max_val = max(max(with_paper_rates or [0]), max(wo_paper_rates or [0])) + 5
min_val = min(min(with_paper_rates or [0]), min(wo_paper_rates or [0])) - 5
ax.set_xlim(min_val, max_val)
ax.set_ylim(min_val, max_val)
ax.set_aspect('equal')

# Improve axis labels
ax.set_xlabel('Success rate without paper context (%)', fontweight='bold')
ax.set_ylabel('Success rate with paper context (%)', fontweight='bold')

# Add grid with improved aesthetics
ax.grid(True, linestyle=':', alpha=0.3, color='gray')

# Add integer ticks
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# Calculate statistics for annotation
avg_diff = sum(diff_rates) / len(diff_rates) if diff_rates else 0
better_with_paper = sum(1 for d in diff_rates if d > 0)
better_without_paper = sum(1 for d in diff_rates if d < 0)
no_difference = sum(1 for d in diff_rates if d == 0)

# Add statistics box with improved aesthetics
stats_text = (
    f"n = {len(snippet_ids)} tasks\n"
    f"Avg. diff: {avg_diff:.2f}pp\n"
    f"Better with paper: {better_with_paper} ({better_with_paper/len(snippet_ids)*100:.1f}%)\n"
    f"Better without: {better_without_paper} ({better_without_paper/len(snippet_ids)*100:.1f}%)"
)
ax.text(0.97, 0.03, stats_text, transform=ax.transAxes, 
        verticalalignment='bottom', horizontalalignment='right', fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9))

# Add a colorbar to show the difference scale
cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, 
                    shrink=0.8, pad=0.1, aspect=30)
cbar.set_label('Difference (pp)', fontsize=12, fontweight='bold')
cbar.ax.tick_params(labelsize=10)

# Adjust layout
plt.tight_layout()

# Save high-quality versions for publication
plt.savefig('snippet_paper_impact_scatter.png', dpi=600, bbox_inches='tight')
plt.savefig('snippet_paper_impact_scatter.pdf', bbox_inches='tight')

print(f"\nPublication-quality scatter plot saved as 'snippet_paper_impact_scatter.png' and .pdf")
print(f"Total snippets analyzed: {len(snippet_ids)}")
print(f"Average difference (with - without paper): {avg_diff:.2f} percentage points")
print(f"Snippets better with paper: {better_with_paper} ({better_with_paper/len(snippet_ids)*100:.1f}% of total)") if snippet_ids else print("No snippets found.")
print(f"Snippets better without paper: {better_without_paper} ({better_without_paper/len(snippet_ids)*100:.1f}% of total)") if snippet_ids else ""
print(f"Snippets with no difference: {no_difference} ({no_difference/len(snippet_ids)*100:.1f}% of total)") if snippet_ids else ""
