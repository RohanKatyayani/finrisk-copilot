# Model Card — FinRisk Copilot

FinRisk Copilot is a portfolio system with three model components served behind one FastAPI
service. This card covers all three: a tabular credit-risk classifier, a LoRA fine-tuned
explanation model, and a retrieval-augmented policy assistant.

**Author:** Rohan Katyayani · **Repository:** https://github.com/RohanKatyayani/finrisk-copilot
**Status:** Portfolio / educational project. **Not approved for production credit decisions.**

---

## 1. Credit Risk Scorer (LightGBM)

### Model details

| Field | Value |
|---|---|
| Architecture | `LGBMClassifier` inside a scikit-learn `Pipeline` |
| Preprocessing | `OneHotEncoder(handle_unknown="ignore")` for categoricals, `StandardScaler` for numerics |
| Key hyperparameters | `learning_rate=0.01`, `max_depth=10`, `num_leaves=30`, `n_estimators=200`, `colsample_bytree=0.8`, `scale_pos_weight=240/560` |
| Random seed | 42 |
| Registry | MLflow Model Registry, `credit_risk_model`, **version 3 → Production** |
| Tracking backend | MLflow with SQLite (`sqlite:///mlflow.db`) |
| Serialization | cloudpickle |

### Training data

Statlog German Credit (see `DATA_CARD.md`). 1,000 rows, 20 features, target `credit_risk`
where **1 = good credit (700 rows)** and **0 = bad credit (300 rows)**. Stratified
80/20 split → 800 train / 200 test (train: 560 good, 240 bad; test: 140 good, 60 bad).

### Evaluation

Measured on the held-out 200-row test set.

| Metric | Value |
|---|---|
| ROC-AUC | **0.768** |
| Accuracy | 0.715 |
| Precision — bad credit (class 0) | 0.519 |
| **Recall — bad credit (class 0)** | **0.683** |
| F1 — bad credit (class 0) | 0.590 |
| Precision — good credit (class 1) | 0.843 |
| Recall — good credit (class 1) | 0.729 |
| F1 — good credit (class 1) | 0.782 |

Reconstructed confusion counts: **41 of 60** bad-credit applicants correctly flagged
(19 missed); **102 of 140** good applicants correctly approved (38 wrongly rejected).

**Baseline context.** Predicting "good" for every applicant yields 0.700 accuracy on this
class distribution. Accuracy alone is therefore close to uninformative here, which is why
ROC-AUC and minority-class recall are reported as the primary metrics.

### Threshold and cost trade-off

The dataset ships with an official cost matrix: misclassifying a bad applicant as good is
**5× more costly** than rejecting a good applicant. Scored on that matrix:

| Model | Missed defaulters | Good rejected | Weighted cost |
|---|---|---|---|
| Earlier version (`scale_pos_weight=2`) | 46 | 6 | (46×5)+(6×1) = **236** |
| Current version (`scale_pos_weight=240/560`) | 19 | 38 | (19×5)+(38×1) = **133** |

The current model has **lower accuracy but 44% lower expected cost**. This is the intended
trade for credit risk: recall on the defaulting class dominates headline accuracy.

### Defects found and corrected

Two related bugs were found by validating assumptions against the data rather than trusting
the code's own documentation. Both are recorded here deliberately.

1. **Inverted label semantics.** The codebase documented `1 = bad credit`, but the target
   column's class balance (700 / 300) matched the dataset's known 70% *good* proportion, and
   a crosstab confirmed that the strongest good-credit indicator (`no checking account`)
   pointed to class 1 at 0.88. Class 1 is **good credit**. The inversion had propagated into
   the API schema, the explanation prompt builder, and the synthetic-explanation generator —
   producing confidently reversed decisions with no crash, no failing test, and green CI.
2. **Inverted class weighting.** `scale_pos_weight=2` was set believing class 1 was the
   minority. Because class 1 is the 70% majority, this amplified the imbalance to roughly
   4.7:1 toward approval and drove bad-credit recall down to 0.233. Corrected to
   `240/560`, which weights toward the true minority.

### Limitations

- **Small dataset.** 1,000 rows total, 200 in test. Metrics carry wide confidence intervals;
  single-split results should not be read as precise.
- **No cross-validation or hyperparameter search.** Hyperparameters were set by hand.
- **No probability calibration.** Predicted probabilities are not calibrated, so the
  reported confidence should not be treated as a true default likelihood.
- **Dataset age and locale.** German applicants, Deutsche Mark amounts, early 1990s. Does not
  transfer to current lending populations.
- **Registry stages are deprecated.** The promotion CLI uses
  `transition_model_version_stage`, deprecated in MLflow 2.9+ in favour of aliases. Kept to
  demonstrate the stage lifecycle; migration to aliases is future work.

### Fairness considerations

**The model uses protected and proxy attributes directly as features:**
`personal_status_sex` (combined marital status and sex), `age`, and `foreign_worker`.

No fairness metrics (demographic parity, equalised odds, error-rate balance across groups)
have been computed, and no mitigation has been applied. In any real lending context this
would be disqualifying: using sex or nationality in a credit decision is unlawful in most
jurisdictions, and age is a regulated attribute. This project keeps the features to remain
faithful to the benchmark dataset, and flags the issue rather than quietly dropping them.

**Recommended before any real use:** remove or audit protected attributes, compute
group-wise error rates, test proxy leakage from remaining features, and document a
threshold-setting policy reviewed by a model-risk function.

---

## 2. Explanation Model (TinyLlama-1.1B + LoRA)

| Field | Value |
|---|---|
| Base model | `TinyLlama-1.1B` |
| Adaptation | LoRA (PEFT), fine-tuned on Google Colab |
| Published weights | [`rohankatyayani/tinyllama-credit-explainer`](https://huggingface.co/rohankatyayani/tinyllama-credit-explainer) |
| Training data | Synthetic bank-tone explanations generated from the German Credit rows |
| Inference | CPU, float32, greedy decoding, `repetition_penalty=1.2`, 120 new tokens |
| Serving | Isolated in a fresh Python subprocess per call |

### Why subprocess isolation

PyTorch deadlocks when a model is loaded inside a forked uvicorn worker on macOS. Each
`/explain` call therefore spawns a short-lived Python process, runs inference, and returns
JSON on stdout. This is a deliberate workaround for local development; in production the
explainer belongs in a separate GPU-backed service.

### Critical limitation — explanations are not attributions

The explainer is conditioned on the applicant's features **and the decision**, then asked to
write prose. It does **not** read the classifier's internal state, and it is **not** derived
from SHAP values or any attribution method. Its output is a *plausible narrative consistent
with the stated decision*, not a faithful account of why the LightGBM model scored the
applicant as it did.

Consequences:

- The explanation may cite factors the model barely weighted, or omit factors it weighted heavily.
- The same profile with a flipped decision will produce confident reasoning for either outcome.
- **This output must not be used as an adverse-action notice or a regulatory explanation.**

A faithful pipeline would compute SHAP attributions from the classifier and condition the
language model on those attributions. That is the natural next iteration.

### Additional notes

- Trained on synthetic text, so tone and structure are consistent but content is generic.
- CPU latency is roughly 10–40 s per explanation; the first call also downloads weights.
- Because the label convention was inverted when the synthetic set was generated, the model
  learned associations between the *decision word* and its prose rather than the integer
  label. Correcting the integer→word mapping at inference restored correct behaviour without
  retraining, verified by inspecting outputs for both decisions on identical profiles.

---

## 3. Policy Assistant (RAG)

| Field | Value |
|---|---|
| Corpus | 3 public AML/KYC documents, 561 chunks (see `DATA_CARD.md`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Index | FAISS `IndexFlatIP` (exact inner-product search) |
| Generation | Groq, `llama-3.1-8b-instant` |
| Default retrieval depth | k = 4 (configurable 1–10) |
| Output | Answer with inline `[1]`, `[2]` citations plus a structured `sources` array with similarity scores |

### Grounding and refusal

The system prompt instructs the model to answer **strictly** from the retrieved passages and
to state that it lacks sufficient information otherwise. Refusal is therefore
**model-mediated rather than threshold-mediated**: the assistant can decline even when
retrieval similarity is moderate (observed refusals with top scores around 0.52–0.56), which
is more robust than a fixed similarity cutoff but depends on instruction-following.

Verified behaviour: in-domain AML/KYC questions return grounded, cited answers; questions
outside the corpus — including capital-adequacy questions, which the corpus does not cover —
are refused rather than answered from the model's pretrained knowledge.

### Limitations

- **Narrow corpus.** Three AML/KYC/CDD documents only. No capital adequacy, no Basel III
  prudential rules, no jurisdiction-specific regulation.
- **No reranking.** Top-k inner-product retrieval with no cross-encoder rerank step.
- **External dependency.** Generation requires a Groq API key; the service returns HTTP 503
  when the FAISS index is absent.
- **Not legal advice.** Retrieved text may be outdated relative to current regulation.

---

## Intended use

Educational demonstration of an end-to-end ML system: tabular modelling, parameter-efficient
fine-tuning, retrieval-augmented generation, experiment tracking, model registry, drift
monitoring, containerisation, and CI.

## Out-of-scope use

Real credit decisions; adverse-action notices; regulatory or compliance advice; any setting
where a person is affected by the output.

---

*Last updated following registry version 3 (Production).*
