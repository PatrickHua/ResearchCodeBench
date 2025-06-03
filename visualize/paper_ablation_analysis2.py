import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json
import numpy as np
# Use serif font for publication-ready look
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
# bold font
plt.rcParams['font.weight'] = 'bold'
# Load the data files
wo_paper_json = json.load(open("outputs/20llms_greedy_wo_paper/2025-05-13-11-43-56/overall_stats.json"))
with_paper_json = json.load(open("outputs/20llms_greedy/2025-05-13-10-05-10/overall_stats.json"))

# Extract data
wo_paper_data = {}
for key, value in wo_paper_json['overall_scores'].items():
    wo_paper_data[key] = {
        'score': value['line_rates']['mean'],
        'pretty_name': value['llm_cfg']['pretty_name']
    }

w_paper_data = {}
for key, value in with_paper_json['overall_scores'].items():
    if key in wo_paper_data:
        w_paper_data[key] = {
            'score': value['line_rates']['mean'],
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

# Create figure and axes
fig, ax = plt.subplots(figsize=(4, 6))

# Create a colormap for the bars
colors = plt.cm.coolwarm(np.interp(improvement, [-10, 10], [0, 1]))
# colors = plt.cm.coolwarm(np.interp(df_sorted['difference'], [-10, 10], [0, 1]))

# Create horizontal bar chart
y_pos = np.arange(len(pretty_names))
bars = ax.barh(y_pos, improvement, color=colors)

# Add vertical line at x=0
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)

# Set y-ticks to pretty names but hide the tick marks
ax.set_yticks(y_pos)
ax.set_yticklabels(pretty_names)
ax.tick_params(axis='y', which='both', length=0)  # Remove y-tick marks
ax.tick_params(axis='x', which='both', length=0)  # Remove x-tick marks

# Add title and labels
ax.set_title('Impact of Paper Context', loc='right', fontweight='bold', fontsize=17)
ax.set_xlabel('Difference in Scaled Pass@1', fontweight='bold')

# Add grid lines
ax.grid(True, alpha=0.3, axis='x')

# Print stats
positive_count = sum(1 for i in improvement if i > 0)
negative_count = sum(1 for i in improvement if i < 0)
neutral_count = sum(1 for i in improvement if i == 0)
avg_improvement = sum(improvement) / len(improvement)

print(f"Improvements statistics:")
print(f"Average improvement: {avg_improvement:.2f} percentage points")
print(f"LLMs with positive improvement: {positive_count}")
print(f"LLMs with negative improvement: {negative_count}")
print(f"LLMs with no improvement: {neutral_count}")

# Add explanation below the plot
plt.figtext(0.55, 0.04, 'Positive values = better with paper  |  Negative values = better without paper', 
           ha='center', fontsize=9)

# Save figure
plt.tight_layout(rect=[0, 0.05, 1, 1])  # Add padding at the bottom for the figtext
plt.savefig("llm_paper_impact.png", dpi=300, bbox_inches='tight')
plt.savefig("llm_paper_impact.pdf", bbox_inches='tight')
plt.savefig("llm_paper_impact.svg", bbox_inches='tight')
# plt.show()
