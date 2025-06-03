


# outputs_main_results/combined_visualization.pdf


python visualize/knowledge_cut_off_ty.py outputs/20llms_greedy/2025-05-13-10-05-10/overall_stats.json model_paper_knowledge_debug_ty.png



cd visualize && python contamination_free_results_ty.py

python contamination_knowledge_cutoff_merged.py


# outputs_main_results/model_line_rates.pdf
python main_results_blue.py



# outputs/error_distribution_pie.pdf
python pie_chart.py



cd ..
# # snippet_paper_impact_scatter.pdf  
# python visualize/paper_ablation_analysis_scatter.py

# python visualize/paper_ablation_analysis2.py
# python visualize/contamination_knowledge_cutoff_merged.py --pdf1_path snippet_paper_impact_scatter.pdf --pdf2_path llm_paper_impact.pdf --output_path llm_paper_impact_merged.pdf

python visualize/paper_ablation_analysis2.py && python visualize/paper_ablation_analysis_scatter.py && python visualize/contamination_knowledge_cutoff_merged.py --pdf1_path snippet_paper_impact_scatter.pdf --pdf2_path llm_paper_impact.pdf --output_path llm_paper_impact_merged.pdf
