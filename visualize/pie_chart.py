import json
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load the JSON data
with open('../outputs_bak2/error_classification/2025-04-24-17-06-43/2025-04-24-17-06-43/error_classification_results.json', 'r') as f:
    data = json.load(f)

# Count occurrences of each category
category_counts = Counter(item['category_name'] for item in data)
# breakpoint()
# Define a threshold for "small" categories (e.g., less than 5% of total)
total_count = sum(category_counts.values())
threshold = 0.05 * total_count

# # Combine small categories into an "Other" category
# other_categories = {k: v for k, v in category_counts.items() if v < threshold}
# if other_categories:
#     # Remove small categories and add a combined "Other" category
#     for k in other_categories:
#         del category_counts[k]
#     category_counts["Other"] = sum(other_categories.values())

# Sort categories by count (descending) for legend order
sorted_items = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
labels, sizes = zip(*sorted_items)
# replace "Logic Error" with "Functional Error"
labels = [label.replace("Logic Error", "Functional Error") for label in labels]
# Use light pastel colors
base_colors = plt.get_cmap('Pastel2').colors
colors = base_colors[:len(labels)]

# Matplotlib RC settings for publication
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14
})

# Explode each slice slightly
explode = [0.04] * len(labels)

# Create a function to display percentages only for slices above a threshold
def autopct_if_large_enough(pct):
    return f'{pct:.0f}%' if pct >= 3 else ''

# Create pie chart
fig, ax = plt.subplots(figsize=(4.9, 3))
wedges, _, autotexts = ax.pie(
    sizes,
    autopct=autopct_if_large_enough,
    startangle=90,
    colors=colors,
    explode=explode,
    wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
    textprops={'fontsize': 12, 'color': 'gray'}
)

# Remove labels from inside the chart, use legend instead
for autotext in autotexts:
    autotext.set_fontsize(11)
    autotext.set_weight('bold')

# Ensure circle shape
ax.axis('equal')

# Add legend with category name and count
total = sum(sizes)
patches = [
    mpatches.Patch(color=colors[i], label=f"{labels[i]}")
    for i in range(len(labels))
]
ax.legend(handles=patches, loc='center left', bbox_to_anchor=(0.98, 0.5), fontsize=11, title="")

# Optional title
# ax.set_title("Error Type Distribution")

# Layout and save
plt.tight_layout()
plt.savefig("outputs/error_distribution_pie.pdf", bbox_inches='tight')  # Uncomment to save
plt.show()
