# linguistic-bias-in-LLM
Linguistic Bias in LLM Outputs


## Execution — Step by Step


```bash
# 0. Setup
pip install -r requirements.txt
export HF_TOKEN="hf_your_token_here"

# 1. Verify (fast — tokenizers only)
python verify_setup.py

# 2. Test visualizations with synthetic data
python generate_mock_data.py
python run_visualization.py

# 3. Run experiments (one model at a time)
python run_single_model.py mistral    # ~45 min
python run_single_model.py llama      # ~50 min
python run_single_model.py qwen       # ~40 min
python run_single_model.py gemma      # ~65 min

# 4. Check integrity
python check_results.py

# 5. Generate final figures
python run_visualization.py
```
