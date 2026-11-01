# Comparative Evaluation of SDG Text Classification Methods

Code and experiments for my NLP project (Sapienza University of Rome, Master's in Engineering in Computer Science and Artificial Intelligence, Natural Language Processing HWp, A.Y. 2025/2026).

The project compares several ways of assigning **UN Sustainable Development Goals (SDGs)** to text under one protocol and one label space (SDGs 1–16, K = 16). It looks at three questions that single-dataset studies cannot answer: how well models **transfer between datasets**, how much **label noise** hurts them, and what changes when moving from **short single-label paragraphs** to **long multi-label documents**.

---

## Systems compared

| Family | Models |
|---|---|
| Traditional baseline | TF-IDF (unigrams + bigrams) with logistic regression (multinomial for Setting A, one-vs-rest with a tuned threshold for Setting B) |
| Transformer encoder | Fine-tuned `RoBERTa-base`, 3 seeds (42, 43, 44); SDGi documents are split into 150-word chunks whose scores are aggregated |
| Existing domain classifier | `sadickam/sdgBERT`, used as released, no further training |
| LoRA-fine-tuned LLMs | `Llama-3.1-8B-Instruct`, `Qwen3.5-9B`, `Gemma-4-12B` (QAT checkpoint); 4-bit base models, trained with [Unsloth](https://github.com/unslothai/unsloth) on a Colab L4 |
| Prompt-only LLMs | The same three models, zero-shot and few-shot, run locally with [LM Studio](https://lmstudio.ai/) with JSON-schema constrained output; plus an exploratory test of ChatGPT, Gemini and DeepSeek on 144 paragraphs |

**How the LoRA models produce scores.** They do not generate text and have no classification head. The prompt asks for one answer word per goal (e.g. *Poverty*, *Health*, *Peace*); a softmax over the next-token logits of the 16 words gives the class scores, and training minimises cross-entropy over those 16 logits. Settings: rank 16, alpha 16, learning rate 2e-4, effective batch size 16, one epoch on 6,000 random OSDG-CD training paragraphs, single run.

---

## Datasets

| Dataset | Used for | Description |
|---|---|---|
| **OSDG Community Dataset (OSDG-CD)**, release of 1 Oct 2023 | **Setting A**, multi-class | Short excerpts (about 96 words), one goal each, validated by volunteers. After the majority-vote filter, 34,933 excerpts remain. The main experiments use agreement ≥ 0.6: 15,984 train / 3,465 dev / 3,566 test, from a document-grouped, SDG-stratified 20-fold split |
| **SDGi Corpus** (UNDP) | **Setting B**, multi-label, and cross-dataset transfer | Long excerpts from national and local sustainability reviews (about 1,218 words) in English, Spanish and French; about 12% of documents have more than one goal. Main evaluation: 965 English test documents; all three languages: 1,342 |

SDG 17 is dropped because OSDG-CD does not contain it. The notebook fetches both automatically (OSDG-CD from Zenodo, SDGi as `UNDP/sdgi-corpus` from Hugging Face). Please cite the original datasets if you reuse them.

---

## Repository structure

```text
├── SDG_classification_MAIN.ipynb   # Main notebook: data prep, TF-IDF, RoBERTa, sdgBERT,
│                                   # prompt-only LLMs, all evaluation tables and figures
├── LLM_fineTuning/                 # Google Colab notebooks for the LoRA models
│                                   # (Qwen3.5-9B, Gemma-4-12B QAT, Llama-3.1-8B, extra scoring)
├── data/                           # OSDG-CD csv (auto-downloaded if missing); SDGi is loaded from the Hugging Face Hub
├── outputs/                        # Generated results: tables, figures, cached predictions
└── readme.md
```

---

## How to reproduce

### Requirements

* Python 3 with: `torch` (CUDA build; RTX 50-series GPUs need CUDA 12.8 or newer), `transformers`, `datasets`, `scikit-learn`, `scipy`, `pandas`, `numpy`, `matplotlib`, `tqdm`, `accelerate`, `sentencepiece`, `safetensors`, `requests`, `pyarrow`
*  The datasets and the Hugging Face models (`roberta-base`, `sadickam/sdgBERT`) are downloaded automatically.
* Environment used for the main notebook: Windows, Python 3.14.7, torch 2.14.1 (CUDA 13.0), NVIDIA GeForce RTX 5060 Laptop GPU (7.9 GiB VRAM).

### Quick start: reuse the cached results

The repository already contains the expensive outputs, so you can reproduce all tables and figures without training anything:

* `outputs/preds/`: cached score matrices of every model (TF-IDF, the three RoBERTa seeds, sdgBERT) and the LoRA score files (`*.npz` and `*_extra.npz`).
* `outputs/llm/`: the cached answers of the prompt-only LLMs, so LM Studio is not needed.
* `outputs/qwen_export.zip`: the exported splits used by the Colab notebooks.

Clone the repository, install the requirements, and run `SDG_classification_MAIN.ipynb` from top to bottom: finished steps are loaded from the cache and only the evaluation, tables and figures are recomputed. RoBERTa checkpoints (`outputs/models/`) are not included because of their size. To recompute a model from scratch, delete its files in `outputs/preds/` (or `outputs/llm/`) and follow the steps below.

### 1. Main notebook (all models except the LoRA training)

Run `SDG_classification_MAIN.ipynb` from top to bottom, with the repository root as the working directory (paths are relative: `data/` and `outputs/`).

* **OSDG-CD** is read from `data/osdg-community-data-v2023-10-01.csv` (tab-separated). If the file is missing, the notebook downloads it from [Zenodo (record 8397907)](https://zenodo.org/records/8397907).
* **SDGi** is loaded directly from the Hugging Face Hub as `UNDP/sdgi-corpus`, so nothing needs to be placed in `data/`.
* Splits use a fixed seed (42); RoBERTa uses seeds 42, 43 and 44.
* Everything slow is cached on disk: score matrices in `outputs/preds/`, RoBERTa checkpoints in `outputs/models/`, LM Studio answers in `outputs/llm/`. After a crash, re-run and finished steps are reloaded. Delete these folders to force recomputation. Tables are written to `outputs/tables/` and figures to `outputs/figs/`.
* If the LoRA score files (step 3) are not present, the registration cell prints a notice and skips them; the rest of the notebook runs normally.

### 2. Prompt-only LLMs (LM Studio) and web chatbots

* Start the LM Studio local server (`http://127.0.0.1:1234/v1`) with 4-bit builds of three models whose ids contain `meta-llama-3.1-8b-instruct`, `qwen/qwen3.5-9b` and `google/gemma-4-12b-qat`. Settings used: temperature 0, JSON-schema constrained output, thinking mode off.
* If the server is off, the notebook uses the answers already cached in `outputs/llm/` and skips the missing ones. Set `RUN_LLM = False` in the LLM config cell to make this explicit.
* The web chatbot comparison (ChatGPT, Gemini, DeepSeek) was done by hand: the notebook generates 18 prompt batches, and the answers I pasted back are stored in the chatbot cell, so nothing has to be queried again to reproduce the table.

### 3. LoRA-fine-tuned LLMs (Google Colab, optional)

The LoRA models are trained outside the main notebook, on a Colab NVIDIA L4 GPU (22 GiB). The main notebook only reads their saved scores.

1. **Export the data.** The file `outputs/qwen_export.zip` is already in the repository; it can be regenerated by running the export cell of the main notebook. It contains the OSDG-CD train/dev/test splits, the SDGi chunks, the indices of the 496-paragraph subsample and the robustness sets.  The same file is used for all three models.
2. **Train.** In `LLM_fineTuning/`, run `qwen.ipynb`, `gemma.ipynb` and `llama.ipynb`, one per Colab session. Run the install cell, restart the session, upload `qwen_export.zip`, set `quick=False` (the default `quick=True` is only a short sanity check) and run all cells. Settings: 6,000 random OSDG-CD training paragraphs, 1 epoch, LoRA rank 16 and alpha 16, learning rate 2e-4, effective batch size 16, seed 42, 4-bit base model. Training took about 44 (Qwen), 71 (Gemma) and 42 (Llama) minutes. Package versions in Colab: Unsloth 2026.10.7, transformers 5.17.0 (pinned to `>=5.0.0,<=5.17.0`), torch 2.11.0 (CUDA 13.0).
3. **Download the scores.** Each notebook produces `qwen_lora_osdg.npz`, `gemma12b_qat_lora_osdg.npz` and `llama31_8b_lora_osdg.npz`, plus an adapter `.zip`.
4. **Robustness scores (Tables on annotation noise and keyword removal).** `llama.ipynb` scores them in its last cell (`llama31_8b_lora_osdg_extra.npz`). For Qwen and Gemma, run `extra_experiment_gemma_qwen.ipynb` with their saved adapters, their two main `.npz` files and `qwen_export.zip`.
5. **Register.** Copy all `.npz` files (main and `_extra`) into `outputs/preds/` (the ones used in the report are already there) and re-run the main notebook from the LoRA registration cell onward. 

---

## Main results

**Setting A, OSDG-CD test (3,566 paragraphs)**

| Model | Accuracy | Macro-F1 |
|---|---|---|
| TF-IDF + LR | 0.862 | 0.820 |
| RoBERTa-base (3 seeds, mean) | 0.893 | 0.857 |
| Qwen3.5-9B LoRA | 0.897 | 0.860 |
| Gemma-4-12B LoRA | 0.886 | 0.847 |
| Llama-3.1-8B LoRA | 0.899 | 0.866 |
| sdgBERT (released model) † | 0.950 | 0.933 |

† sdgBERT was trained on OSDG data, so its training set may overlap this test split. Its OSDG-CD numbers are not a fair comparison.

**Setting B, English SDGi test (965 documents), micro-F1 with a validation-tuned threshold**

| Trained on SDGi | Micro-F1 | | Trained on OSDG-CD (transfer) | Micro-F1 |
|---|---|---|---|---|
| TF-IDF + LR | **0.751** | | Qwen3.5-9B LoRA | 0.678 |
| RoBERTa-base | 0.736 | | Gemma-4-12B LoRA | 0.676 |
| | | | Llama-3.1-8B LoRA | 0.640 |
| | | | TF-IDF + LR / RoBERTa / sdgBERT | 0.607 / 0.605 / 0.599 |

Prompt-only LLMs on the 496-paragraph OSDG-CD subsample reach 0.49–0.76 accuracy zero-shot and 0.56–0.79 few-shot, against 0.83–0.85 for the LoRA models on the same paragraphs. Full tables are in the report and in `outputs/tables/`.

---

## Key findings

* **Simple baselines are strong.** RoBERTa and the LoRA models beat TF-IDF + logistic regression on OSDG-CD by about 3–4 points, but on SDGi, TF-IDF trained in-domain (0.751 micro-F1) is slightly ahead of RoBERTa (0.736).
* **Prompting alone is not enough.** Zero- and few-shot LLMs stay below the TF-IDF baseline. LoRA adds roughly 9 to 26 accuracy points over the same untrained model scored the same way, and it brings the three LLMs much closer to each other.
* **Domain-specific does not mean better out of domain.** sdgBERT is the best model on OSDG-CD (with the overlap caveat) but no better than OSDG-trained TF-IDF on SDGi.
* **Transfer to long multi-label documents.** Among models trained only on OSDG-CD, the LoRA models transfer best, with high precision but lower recall: they often miss the secondary goals of a document. They also keep their performance on Spanish and French documents, while the English-only models drop sharply.
* **Label noise matters.** Accuracy falls by 29 to 40 points between high-agreement (≥ 0.8) and low-agreement (< 0.2) OSDG-CD items, so a large share of the errors comes from ambiguous annotations.
* **Goal difficulty.** SDGs 8, 9 and 10 are the hardest and are often confused with each other and with SDG 1; SDG 11 is the most over-predicted goal on SDGi.

---

## Limitations

* LoRA models and most baselines are single runs; only RoBERTa has three seeds. Differences of about one point should not be read as a ranking.
* The LoRA models were trained on 6,000 of the 15,984 training paragraphs (compute budget), while RoBERTa used the full split.
* The OSDG-CD split is my own, so numbers are not directly comparable with other papers. The SDGi test set is the official one and is harder than a random sample.
* Prompt-only LLMs and the chatbot test run on fixed subsamples (496 paragraphs, 600 documents, 144 paragraphs); the chatbot results are exploratory and the exact model versions were not recorded.
* SDGi chunk labels are weakly supervised (each chunk inherits its document's labels), and decision thresholds are tuned on English data only.

-