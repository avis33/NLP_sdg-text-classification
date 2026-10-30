# Comparative Evaluation of SDG Text Classification Methods

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains the code, data preprocessing pipelines, and experimental evaluations for my Natural Language Processing (NLP) university project investigating automated text classification by the **United Nations Sustainable Development Goals (SDGs)**. 

The project provides a rigorous, side-by-side comparative evaluation of diverse NLP paradigms under a unified 16-class label space ($K=16$, SDGs 1–16), addressing critical challenges in the field such as **cross-dataset generalisation**, **label noise robustness**, and the transition from **single-label short paragraphs** to **multi-label long policy documents**.

---

## 📊 Evaluated Models & Paradigms

1. **Traditional Baseline:** TF-IDF unigrams & bigrams + Logistic Regression (One-Vs-Rest for multi-label).
2. **Transformer Encoders:** Fine-tuned `RoBERTa-base` (with chunk aggregation for long documents).
3. **Domain-Specific Classifiers:** Off-the-shelf `sadickam/sdgBERT` (zero-shot evaluation).
4. **Parameter-Efficient LLMs:** LoRA-adapted open-weight models (`Qwen3.5-9B`, `Gemma-4-12B-QAT`, `Llama-3.1-8B-Instruct`) trained via **Unsloth** using next-token logit scoring over label words.
5. **Prompt-Only LLMs:** Zero- and few-shot prompting via **LM Studio** with JSON-schema constrained generation, alongside exploratory tests with hosted web chatbots (ChatGPT, Gemini, DeepSeek).

---

## 📂 Datasets

Experiments are conducted across two major public benchmarks restricted to a shared label space of **SDGs 1--16** (excluding SDG 17):
* **OSDG Community Dataset (OSDG-CD):** Short text excerpts ($\sim$96 words) labelled via volunteer community assessments. Used for **Setting A (multi-class)** classification on paragraphs.
* **SDGi Corpus:** Long governmental and local sustainability reviews ($\sim$1,218 words) developed by the UNDP. Used for **Setting B (multi-label)** document classification across English, Spanish, and French.

---

## 🛠️ Repository Structure

```text
├── data/                         # Directory for raw datasets (OSDG-CD CSV, SDGi corpus)
├── outputs/
│   ├── figs/                     # Exported evaluation figures (confusion matrices, co-occurrence heatmaps)
│   └── tables/                   # Exported CSV and LaTeX table files (.csv, .tex)
├── SDG_classification_MAIN.ipynb  # Main executable Jupyter Notebook containing all pipelines
└── README.md                     # Project documentation
```

---


## 📈 Key Findings

* **Simplicity vs. Complexity:** While transformer encoders (`RoBERTa`) and LoRA-adapted LLMs outperform linear baselines in-domain, simpler TF-IDF + Logistic Regression models remain highly competitive—particularly when trained directly on target-domain documents.
* **Label Structure & Multi-Label Challenge:** Models trained purely on single-label OSDG paragraphs maintain high precision but suffer from lower recall on multi-label SDGi documents, missing secondary goals.
* **Cross-Lingual Resilience:** LoRA-adapted multilingual LLMs (like Qwen and Gemma) maintain robust performance on Spanish and French test documents despite being fine-tuned exclusively on English training text.
* **Label Noise Sensitivity:** Model accuracy drops by 29 to 40 percentage points between high-agreement ($\geq 0.8$) and low-agreement ($<0.2$) OSDG test items, highlighting that annotation ambiguity accounts for a significant portion of classifier errors.

