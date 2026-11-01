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

SDG 17 is dropped because OSDG-CD does not contain it. The datasets are not redistributed in a way that replaces their own licences: please obtain them from their original sources (OSDG-CD on Zenodo, SDGi as `UNDP/sdgi-corpus` on Hugging Face) and cite them.

---

## Repository structure

```text
├── SDG_classification_MAIN.ipynb   # Main notebook: data prep, TF-IDF, RoBERTa, sdgBERT,
│                                   # prompt-only LLMs, all evaluation tables and figures
├── LLM_fineTuning/                 # Google Colab notebooks for the LoRA models
│                                   # (Qwen3.5-9B, Gemma-4-12B QAT, Llama-3.1-8B, extra scoring)
├── data/                           # Raw datasets (OSDG-CD, SDGi)
├── outputs/                        # Generated results: tables, figures, cached predictions
└── readme.md
```

---

## How to reproduce

1. **Data.** Put OSDG-CD and SDGi into `data/` (paths are set in the first cells of the main notebook).
2. **Main notebook.** Open `SDG_classification_MAIN.ipynb` and run it top to bottom. Slow steps (RoBERTa training, off-the-shelf predictions, LLM answers) are cached in `outputs/`, so re-running is fast and resumable. It was developed on a laptop with an RTX 5060 GPU (about 8 GB VRAM), in bf16.
3. **Prompt-only LLMs.** Start the LM Studio local server (OpenAI-compatible, `http://127.0.0.1:1234/v1`) with the three 4-bit models loaded before running the prompting section.
4. **LoRA models.** The notebook exports a data folder (`qwen_export.zip`) used by the Colab notebooks in `LLM_fineTuning/`. Run them on an L4 GPU, download the resulting `.npz` score files into `outputs/preds/`, and re-run the notebook from the registration cell onward. The trained adapters are not stored in this repository.

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

---

## Acknowledgements and citation

Datasets: OSDG Community Dataset (OSDG / UN Development Programme / PPMI) and the SDGi Corpus (UNDP, Skrynnyk et al., 2024). Models: `sadickam/sdgBERT`, RoBERTa (Liu et al., 2019), Llama 3.1, Qwen3.5 and Gemma 4. Tooling: scikit-learn, Hugging Face Transformers and Datasets, PyTorch, Unsloth, LM Studio. The full reference list is in the project report.

AI tools were used to help write and edit parts of the code and the report; I am responsible for all content and results.
