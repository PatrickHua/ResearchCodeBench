import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json
import numpy as np
# Use serif font for publication-ready look
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 6

wo_paper_json = json.load(open("outputs/20llms_greedy_wo_paper/2025-05-13-11-43-56/overall_stats.json"))
with_paper_json = json.load(open("outputs/20llms_greedy/2025-05-13-10-05-10/overall_stats.json"))


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


# Extract data from dictionaries
with_paper = []
without_paper = []
pretty_names = []

for key in w_paper_data:
    with_paper.append(w_paper_data[key]['score'])  # Convert to percentage
    without_paper.append(wo_paper_data[key]['score'])  # Convert to percentage
    pretty_names.append(w_paper_data[key]['pretty_name'])

# Axis limits with 5% margin
min_val = min(with_paper + without_paper)
max_val = max(with_paper + without_paper)
margin = (max_val - min_val) * 0.05
axis_min = min_val - margin
axis_max = max_val + margin

# Plot setup
fig, ax = plt.subplots(figsize=(6, 4))
# Identity line
ax.plot([axis_min, axis_max], [axis_min, axis_max], linestyle='--', color='grey', linewidth=1)

# Create a color dictionary for pretty names
import matplotlib.cm as cm
colors = cm.rainbow(np.linspace(0, 1, len(pretty_names)))
color_dict = {name: colors[i] for i, name in enumerate(pretty_names)}

# Plot points and drop lines with consistent colors per model
for x, y, name in zip(with_paper, without_paper, pretty_names):
    color = color_dict[name]
    ax.scatter(x, y, marker='x', color=color, alpha=0.8, s=30)
    ax.plot([x, x], [y, x], linestyle=':', color=color, linewidth=1.2, alpha=0.8)
    ax.text(x, y, name, fontsize=5, alpha=0.9)

# Legend for models
legend_elements = [
    Line2D([0], [0], marker='x', color=color_dict[name], linestyle=':', 
           label=name, markersize=5, linewidth=1.2, alpha=0.8)
    for name in pretty_names
]
# ax.legend(handles=legend_elements, fontsize=6, title='Models', framealpha=0.8)

# Labels and title
ax.set_xlim(axis_min, axis_max)
ax.set_ylim(axis_min, axis_max)
ax.set_xlabel("Original (with paper) Mean Success Rate (%)", fontsize=8)
ax.set_ylabel("Without Paper Mean Success Rate (%)", fontsize=8)
ax.set_title("Performance Drop from Removing Paper Context", fontsize=9)

plt.tight_layout()
plt.savefig("visualize/paper_ablation.pdf", bbox_inches='tight')
plt.savefig("visualize/paper_ablation.png", bbox_inches='tight', dpi=300)
plt.show()
