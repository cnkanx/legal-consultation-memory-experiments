# Legal Consultation Memory Experiments

Code, frozen synthetic data, and result files for experiments on legal-consultation fact updates and versioned memory.

## Scope

The repository contains:

- Event-level versioned memory code.
- Event extraction and normalization utilities.
- Mem0 baseline integration.
- Ablation experiments.
- J1-Bench legal-consultation integration code.
- 20 frozen synthetic legal-consultation cases.
- Controlled and external result files.

The upstream J1-Bench source code and original J1-Eval dataset are not redistributed here. Use the upstream project:

<https://github.com/FudanDISC/J1Bench>

## Layout

```text
code/
dataset/
results/
  controlled/
  j1_external/
results_summary/
```

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

API credentials must be provided through local environment variables:

```powershell
$env:J1_EVAL_API_KEY = "<your-api-key>"
$env:J1_EVAL_API_BASE = "https://api.deepseek.com/v1"
$env:J1_EVAL_MODEL_NAME = "deepseek-chat"
$env:J1_EVAL_JUDGE_MODEL = "deepseek-chat"
```

Do not commit API keys or local runtime databases.

## Result Files

- `results/controlled/`: Oracle, Mem0, DeepSeek extraction, and ablation outputs.
- `results/j1_external/`: final J1-Bench scores for three runs.
- `results_summary/`: CSV summaries for tables and plots.

## Reproducibility Notes

- Oracle results use human-annotated structured events and test the memory mechanism only.
- DeepSeek results include automatic extraction errors.
- Mem0 was run with `mem0ai==2.2.1`, `BAAI/bge-small-zh-v1.5`, a local Qdrant store, and Top-k 10.
- Strict exact-match and normalized rescoring results are retained separately.
- The J1-Bench integration keeps the upstream simulator and evaluator unchanged.

