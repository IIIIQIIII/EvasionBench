# EvasionBench

A large-scale benchmark for detecting managerial evasion in earnings call Q&A.

## Overview

EvasionBench introduces a three-level evasion taxonomy (**direct**, **intermediate**, **fully evasive**) and a Multi-Model Consensus (MMC) annotation framework using frontier LLMs. Our benchmark includes 84K balanced training samples and a 1K gold-standard evaluation set.

**Eva-4B**, a fine-tuned Qwen3-4B model, achieves **84.9% Macro-F1**, outperforming larger frontier models including Claude Opus 4.5 and GPT-5.2.

## Project Structure

```
EvasionBench/
├── index.html                 # Project page
├── README.md                  # This file
├── assets/                    # Static assets for project page
│   ├── favicon.svg            # Site favicon (pink background + "E")
│   ├── mmc_pipe.svg           # MMC pipeline diagram
│   ├── top5_performance.svg   # Top 5 model performance chart
│   ├── training_pipeline_pastel.svg
│   ├── eva_4b_ablation_study_pastel.svg
│   ├── training_loss_curve_pastel.png
│   ├── eva_4b_confusion_matrix_pastel.png
│   ├── judge_label_distribution_pastel.png
│   └── position_bias_analysis_pastel.png
├── data/                      # EvasionBench data samples
│   └── evasionbench_test_cases_6samples.csv
└── prompts/                   # Prompt templates
    ├── evasion_rasiah_fine_tuning_minimalist.txt
    └── evasion_rasiah_with_reason_prompt_template.txt
```

### `/data`

Contains EvasionBench data samples for evaluation and demonstration.

| File | Description |
|------|-------------|
| `evasionbench_test_cases_6samples.csv` | Sample Q&A pairs with gold labels |

### `/prompts`

Contains prompt templates used for evasion classification.

| File | Description |
|------|-------------|
| `evasion_rasiah_fine_tuning_minimalist.txt` | Minimalist prompt for fine-tuning |
| `evasion_rasiah_with_reason_prompt_template.txt` | Full prompt with reasoning |

## Evasion Taxonomy

| Label | Definition |
|-------|------------|
| **Direct** | The core question is directly and explicitly answered |
| **Intermediate** | The response provides related context but sidesteps the specific core |
| **Fully Evasive** | The question is ignored, refused, or the response is entirely off-topic |

## Model Performance

| Rank | Model | Macro-F1 |
|------|-------|----------|
| 1 | **Eva-4B (Full)** | **84.9%** |
| 2 | Gemini 3 Flash | 84.6% |
| 3 | Claude Opus 4.5 | 84.4% |
| 4 | GLM-4.7 | 82.9% |
| 5 | Eva-4B (Consensus) | 81.4% |

## Links

- [Project Page](https://iiiiqiiii.github.io/EvasionBench)
- [Model on HuggingFace](https://huggingface.co/FutureMa/Eva-4B)
- [Paper (arXiv)](https://arxiv.org/abs/2602.xxxxx)

## Citation

```bibtex
@article{evasionbench2026,
  title={EvasionBench: A Large-Scale Benchmark for Detecting Managerial Evasion in Earnings Call Q&A},
  author={...},
  journal={arXiv preprint arXiv:2602.xxxxx},
  year={2026}
}
```

## License

Apache 2.0
