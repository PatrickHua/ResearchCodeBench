import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

# Set up publication-quality aesthetics for the entire figure
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
wo_paper_json = json.load(open("../outputs/20llms_greedy_wo_paper/2025-05-13-11-43-56/overall_stats.json"))
with_paper_json = json.load(open("../outputs/20llms_greedy/2025-05-13-10-05-10/overall_stats.json"))

# Create the main figure with two subplots side by side
fig = plt.figure(figsize=(14, 6.5))
# Use GridSpec for more control over the layout
gs = gridspec.GridSpec(1, 2, width_ratios=[1.1, 1])

# Add subplot letters
fig.text(0.01, 0.98, "a", fontsize=16, fontweight="bold")
fig.text(0.52, 0.98, "b", fontsize=16, fontweight="bold")

# ======================== Left plot (Task-level scatter) ========================
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
np.random.seed(42)  # Set seed for reproducibility
wo_paper_rates_jittered = [x + np.random.uniform(-jitter, jitter) for x in wo_paper_rates]
with_paper_rates_jittered = [y + np.random.uniform(-jitter, jitter) for y in with_paper_rates]

# Create color coding based on difference
diff_rates = [w - wo for w, wo in zip(with_paper_rates, wo_paper_rates)]
cmap = plt.cm.RdBu_r
norm = mpl.colors.Normalize(vmin=-20, vmax=20)
colors = [cmap(norm(diff)) for diff in diff_rates]

# Create the task scatter plot
ax1 = plt.subplot(gs[0])

# Add reference lines
ax1.axline((0, 0), slope=1, color='black', linestyle='--', alpha=0.5, zorder=1, linewidth=1.2)
ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.2, zorder=1)
ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.2, zorder=1)

# Create the scatter plot with color coding
scatter = ax1.scatter(wo_paper_rates_jittered, with_paper_rates_jittered, 
                     c=colors, alpha=0.75, s=70, edgecolors='dimgray', 
                     linewidth=0.5, zorder=2)

# Set the axes to be square and with same limits
max_val = max(max(with_paper_rates or [0]), max(wo_paper_rates or [0])) + 5
min_val = min(min(with_paper_rates or [0]), min(wo_paper_rates or [0])) - 5
ax1.set_xlim(min_val, max_val)
ax1.set_ylim(min_val, max_val)
ax1.set_aspect('equal')

# Improve axis labels
ax1.set_xlabel('Success rate without paper context (%)', fontweight='bold')
ax1.set_ylabel('Success rate with paper context (%)', fontweight='bold')

# Add grid with improved aesthetics
ax1.grid(True, linestyle=':', alpha=0.3, color='gray')

# Add integer ticks
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

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
ax1.text(0.97, 0.03, stats_text, transform=ax1.transAxes, 
        verticalalignment='bottom', horizontalalignment='right', fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9))

# Add a colorbar to show the difference scale
cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax1, 
                    shrink=0.8, pad=0.02, aspect=30)
cbar.set_label('Difference (pp)', fontsize=12, fontweight='bold')
cbar.ax.tick_params(labelsize=10)

# ======================== Right plot (LLM comparison) ========================
# Extract data for the LLM comparison
wo_paper_data = {}
for key, value in wo_paper_json['overall_scores'].items():
    wo_paper_data[key] = {
        'score': value['line_rates']['mean'],  # Keep original scale
        'pretty_name': value['llm_cfg']['pretty_name']
    }

w_paper_data = {}
for key, value in with_paper_json['overall_scores'].items():
    if key in wo_paper_data:
        w_paper_data[key] = {
            'score': value['line_rates']['mean'],  # Keep original scale
            'pretty_name': value['llm_cfg']['pretty_name']
        }

# Create lists for the plot
llm_ids = []
pretty_names = []
with_paper_scores = []
wo_paper_scores = []
improvements = []

# Calculate improvements
for key in w_paper_data:
    llm_ids.append(key)
    pretty_names.append(w_paper_data[key]['pretty_name'])
    with_paper_score = w_paper_data[key]['score']
    wo_paper_score = wo_paper_data[key]['score']
    with_paper_scores.append(with_paper_score)
    wo_paper_scores.append(wo_paper_score)
    improvements.append(with_paper_score - wo_paper_score)

# Create a dictionary with all data
data = list(zip(llm_ids, pretty_names, with_paper_scores, wo_paper_scores, improvements))

# Sort by improvement (descending)
sorted_data = sorted(data, key=lambda x: x[4], reverse=True)

# Unpack sorted data
llm_ids = [item[0] for item in sorted_data]
pretty_names = [item[1] for item in sorted_data]
with_paper = [item[2] for item in sorted_data]
without_paper = [item[3] for item in sorted_data]
improvement = [item[4] for item in sorted_data]

# Create the LLM comparison plot
ax2 = plt.subplot(gs[1])

# Create a colormap for the bars using the same colormap as the scatter plot
colors = plt.cm.RdBu_r(np.interp(improvement, [-20, 20], [0, 1]))

# Create horizontal bar chart
y_pos = np.arange(len(pretty_names))
bars = ax2.barh(y_pos, improvement, color=colors, height=0.7)

# Add vertical line at x=0
ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.7)

# Set y-ticks to pretty names with shortened labels if needed
ax2.set_yticks(y_pos)
# Format pretty names to fit better
formatted_names = [name if len(name) < 25 else name[:22]+'...' for name in pretty_names]
ax2.set_yticklabels(formatted_names)

# Add labels
ax2.set_xlabel('Performance difference (percentage points)', fontweight='bold')

# Add grid lines
ax2.grid(True, alpha=0.3, axis='x', linestyle=':')

# Add a title and annotation
# Calculate statistics
positive_count = sum(1 for i in improvement if i > 0)
negative_count = sum(1 for i in improvement if i < 0)
neutral_count = sum(1 for i in improvement if i == 0)
avg_improvement = sum(improvement) / len(improvement)

stats_text = (
    f"n = {len(llm_ids)} models\n"
    f"Avg. diff: {avg_improvement:.2f}pp\n"
    f"Better with paper: {positive_count} ({positive_count/len(llm_ids)*100:.1f}%)\n"
    f"Better without: {negative_count} ({negative_count/len(llm_ids)*100:.1f}%)"
)
ax2.text(0.97, 0.03, stats_text, transform=ax2.transAxes,
         verticalalignment='bottom', horizontalalignment='right', fontsize=11,
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9))

# Set x-axis limits to be symmetric around 0 or at least cover the data range
max_abs_improvement = max(abs(min(improvement)), abs(max(improvement)))
ax2.set_xlim(-max_abs_improvement - 1, max_abs_improvement + 1)

# Add a figure title
plt.suptitle('Impact of Paper Context on Code Generation Performance', 
             fontsize=16, fontweight='bold', y=0.98)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.94])  # Leave space for the super title

# Save high-quality versions for publication
plt.savefig('paper_ablation_merged.png', dpi=600, bbox_inches='tight')
plt.savefig('paper_ablation_merged.pdf', bbox_inches='tight')

print(f"\nMerged figure saved as 'paper_ablation_merged.png' and .pdf")
print(f"Task analysis: {len(snippet_ids)} tasks, average improvement: {avg_diff:.2f}pp")
print(f"Model analysis: {len(llm_ids)} models, average improvement: {avg_improvement:.2f}pp")
