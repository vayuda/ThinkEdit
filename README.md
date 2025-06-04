# ThinkEdit

This is an unofficial spin off repository for the paper: [**ThinkEdit: Interpretable Weight Editing to Mitigate Overly Short Thinking in Reasoning Models**](https://arxiv.org/abs/2503.22048)[[project website](https://lilywenglab.github.io/ThinkEdit/)]. 

The purpose is to extend the original codebase to support newer reasoning models and integrate vllm for faster steering experiments.

### Supported Models

* deepseek-qwen-1.5b
* deepseek-llama3-8b
* deepseek-qwen-14b
* qwen3-1.7b

To add additional models, simply add an identifier and a local or huggingface path in `utils.py`

### Supported Datasets

* openai/gsm8k
* furonghuang-lab/Easy2Hard-Bench gsm8k split

To add additional datasets, follow the examples found in `utils.py`. Identify the name,split and question/answer keys used.



## Run Steering Experiments

To run steering experiments, follow these steps:

### 1. Generate Model Output

Run `generate_responses.py` to generate model output for a given dataset. The script takes the following arguments:

* `--model`: The name of the model to use (e.g. `deepseek-qwen-1.5b`)
* `--dataset`: The name of the dataset to use (e.g. `gsm8k`)
* `--batch_size`: The batch size to use for generation (default: 1)
* `--tp`: The tensor parallel size to use for generation (default: 1)

Example:
```bash
python3 generate_responses.py --model deepseek-qwen-1.5b --dataset gsm8k --batch_size 32 --tp 2
```

### 2. Extract Steering Vector

Run `extract_tl.py` to extract the steering vector for a given model and dataset. The script takes the following arguments:

* `--model`: The name of the model to use (e.g. `deepseek-qwen-1.5b`)
* `--control`: The type of control to use (e.g. `attn` or `mlp`)

Example:
```bash
python3 extract_tl.py --model deepseek-qwen-1.5b --control attn
```

### 3. Run Steering Experiments

Run `run_mlp_steering_experiments.sh` to run steering experiments by intervening after each MLP layer for a given model and dataset. The script takes the following arguments:

* `model`: The name of the model to use (e.g. `deepseek-qwen-1.5b`)
* `dataset`: The name of the dataset to use (e.g. `gsm8k`)
* `device`: The CUDA GPU number(s) that you wish to use for running the experiments

Example:
```bash
bash run_mlp_steering_experiments.sh deepseek-qwen-1.5b gsm8k
```
The  script `run_mlp_steering_experiments.sh` works in a similar manner, but intervenes after every attention layer.

### 4. Plot Steering Results

Run `plot_steering.py` to plot the steering results for a given directory of CSV files. The script takes the following arguments:

* `dir_path`: The path to the directory containing the CSV files

Example:
```bash
python3 plot_steering.py results/qwen3-1.7b_steering_results
```

This will create two summary figures:

* `combined_thinking_length_vs_steering_strength.png`
* `combined_accuracy_vs_steering_strength.png`

inside the specified directory.

Note: Make sure to update the `model_dict` and `DATASET_MAP` variables in `generate_responses.py` and `extract_tl.py` to include the models and datasets you want to support.

## Cite the original work:

Chung-En Sun, Ge Yan, Tsui-Wei Weng, "ThinkEdit: Interpretable Weight Editing to Mitigate Overly Short Thinking in Reasoning Models", arxiv preprint

```bibtex
@article{ThinkEdit,
   title={ThinkEdit: Interpretable Weight Editing to Mitigate Overly Short Thinking in Reasoning Models},
   author={Chung-En Sun, Ge Yan, Tsui-Wei Weng},
   journal={arXiv},
   year={2025}
}
```



