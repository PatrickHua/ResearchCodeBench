# ResearchCodeBench



---

## Environment Setup

1. **Create and activate the Conda environment**  
   ```bash
   conda env create -f environment.yml
   conda activate researchcodebench
   ```

2. **Run the sanity check**  
   ```bash
   python sanity_check_tests.py
   ```  
   This script runs unit tests for each codebase. On a correct install, you should see timings like:

   | Repository ID                  | Test Time (s) |
   | ------------------------------ | ------------: |
   | advantage-alignment            |         0.022 |
   | Diff-Transformer               |         0.037 |
   | DiffusionDPO                   |         0.004 |
   | DyT                            |         0.016 |
   | eomt                           |         0.390 |
   | fractalgen                     |         0.026 |
   | GMFlow                         |         0.107 |
   | GPS                            |         0.013 |
   | grid-cell-conformal-isometry   |         0.030 |
   | hyla                           |         1.914 |
   | LEN                            |         0.051 |
   | llm-sci-use                    |         2.342 |
   | minp                           |         0.004 |
   | OptimalSteps                   |         0.288 |
   | REPA-E                         |        19.240 |
   | schedule_free                  |         0.454 |
   | semanticist                    |         0.023 |
   | SISS                           |         0.016 |
   | TabDiff                        |         0.014 |
   | Tanh-Init                      |         0.001 |

---

## API Keys

Before running any models, export your API keys:

```bash
export OPENAI_API_KEY="…"
export GOOGLE_API_KEY="…"
export XAI_API_KEY="…"
export DEEPSEEK_API_KEY="…"
export OPENROUTER_API_KEY="…"
```

---

## Running the Benchmark

On any CPU-only machine:

```bash
sh run_greedy.sh
```

- By default this will evaluate all 32 models from the paper.  
- To target a single model, edit `run_greedy.sh`.

Results are saved to:
```
outputs/20llms_greedy/<YYYY-MM-DD-HH-MM-SS>/
```

---

## Generating the Main Figure

To reproduce Figure 2 (Scaled Pass@1 results), a JSON file produced by the command above is provided.

1. Ensure you have the JSON file:
   ```
   outputs/20llms_greedy/2025-05-12-17-13-20/overall_stats.json
   ```
2. Run the plotting script:
   ```bash
   python visualize/main_results_blue.py --json_path outputs/20llms_greedy/2025-05-12-17-13-20/overall_stats.json
   ```
3. The plots will be saved as:
   ```
   outputs_main_results/model_line_rates.png
   outputs_main_results/model_line_rates.pdf
   ```
