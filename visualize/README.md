# Visualization Scripts

This directory contains scripts for visualizing various aspects of the paper2code project.

## Knowledge Cutoff Visualization

The `knowledge_cut_off_sm.py` script creates a visualization showing the relationship between language model knowledge cutoff dates and research paper implementation dates.

### Usage

```
python knowledge_cut_off_sm.py <path_to_json_file> [output_path]
```

Example:
```
python knowledge_cut_off_sm.py results.json knowledge_cutoff_visualization.png
```

### Features

- Automatically extracts model information from `overall_stats.json` in the outputs directory
- Shows knowledge cutoff dates for each model categorized by developer
- Displays first and last commit dates for research papers
- Calculates the percentage of models that have knowledge cutoff dates before each paper's commit dates
- Creates both PNG and PDF outputs for high-quality visualizations
- Prints a summary of the analysis results

### Output

The script generates:
1. A horizontal visualization with papers on the x-axis and a timeline on the y-axis
2. PNG and PDF versions of the visualization
3. A summary analysis in the console showing:
   - Total number of models with knowledge cutoff dates
   - Total number of papers analyzed
   - Unique knowledge cutoff dates and model counts
   - Average percentage of models with knowledge before first commit
   - Average percentage of models with knowledge before last commit

### Configuration

The script automatically detects model information from:
1. The `overall_stats.json` file in the outputs directory 
2. If not found, it falls back to extracting model info from the provided JSON file

This allows for accurate representation of model knowledge cutoff dates based on the actual models used in the evaluation.

### Exclusions

- **Mistral AI models**: Models from Mistral AI are excluded from the visualization because their knowledge cutoff dates are unclear. The script will print a message when these models are skipped.
- **Amazon/Claude models**: Models from Amazon (including Claude models) are excluded from the visualization because their knowledge cutoff dates are unclear. The script will print a message when these models are skipped. 