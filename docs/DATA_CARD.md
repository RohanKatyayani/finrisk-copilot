# Data Card — FinRisk Copilot

Two distinct datasets power this project: a tabular credit dataset for the risk scorer and
explanation generator, and a small document corpus for the retrieval-augmented policy
assistant.

---

## 1. Statlog German Credit Data

### Provenance

| Field | Value |
|---|---|
| Name | Statlog (German Credit Data) |
| Original source | UCI Machine Learning Repository |
| Donor | Prof. Hans Hofmann, Universität Hamburg |
| Period | Early 1990s |
| Population | German bank credit applicants |
| Currency | Deutsche Mark (DM) |
| Instances | 1,000 |
| Features | 20 (13 categorical, 7 numeric) |
| Repository path | `data/interim/german_credit.csv` (tracked in git, ~220 KB) |

### Target variable

Column `credit_risk`:

| Value | Meaning | Count | Share |
|---|---|---|---|
| `1` | Good credit | 700 | 70% |
| `0` | Bad credit | 300 | 30% |

**Encoding note.** The original UCI distribution encodes the target as `1 = good`, `2 = bad`.
This project's preprocessing maps it to `1 = good`, `0 = bad`. That convention was verified
empirically rather than assumed: the 700/300 split matches the dataset's documented 70% good
proportion, and a crosstab of `status` against the target showed `no checking account` — the
dataset's strongest good-credit signal — associating with class 1 at 0.88. Earlier code in
this repository documented the opposite convention and was corrected.

### Official cost matrix

The dataset ships with an asymmetric cost matrix reflecting real lending economics:

| | Predicted good | Predicted bad |
|---|---|---|
| **Actually good** | 0 | 1 |
| **Actually bad** | **5** | 0 |

Approving an applicant who defaults is **five times** as costly as rejecting a
creditworthy one. This is the stated justification for optimising minority-class recall over
accuracy in the model card.

### Features

Categorical attributes are stored as the dataset's original `Axxx` codes and one-hot encoded
at training time. The UI maps codes to human-readable labels for display only.

| Feature | Type | Notes |
|---|---|---|
| `status` | categorical | Checking account status (A11–A14) |
| `duration` | numeric | Loan duration in months |
| `credit_history` | categorical | A30–A34 |
| `purpose` | categorical | A40–A410 |
| `amount` | numeric | Credit amount in DM |
| `savings` | categorical | A61–A65 |
| `employment_duration` | categorical | A71–A75 |
| `installment_rate` | numeric | 1–4, percent of disposable income |
| `personal_status_sex` | categorical | **Protected** — combines marital status and sex (A91–A95) |
| `other_debtors` | categorical | None / co-applicant / guarantor |
| `present_residence` | numeric | Years at current residence |
| `property` | categorical | A121–A124 |
| `age` | numeric | **Regulated attribute** — years |
| `other_installment_plans` | categorical | Bank / stores / none |
| `housing` | categorical | Own / rent / free |
| `number_credits` | numeric | Existing credits at this bank |
| `job` | categorical | A171–A174 |
| `people_liable` | numeric | Dependents |
| `telephone` | categorical | Registered / none |
| `foreign_worker` | categorical | **Protected** — yes / no |

### Splits

Stratified 80/20 with `random_state=42`:

| Split | Rows | Good | Bad |
|---|---|---|---|
| Train | 800 | 560 | 240 |
| Test | 200 | 140 | 60 |

After one-hot encoding, the model sees 57 features.

### Known issues and ethical considerations

- **Protected attributes are present and used.** `personal_status_sex` and `foreign_worker`
  encode sex and nationality; `age` is regulated in most lending jurisdictions. They are
  retained for fidelity to the benchmark and flagged rather than silently dropped. No fairness
  audit has been performed — see the model card.
- **Proxy leakage.** Even with protected columns removed, features such as `housing`,
  `property`, and `job` can act as proxies for protected characteristics.
- **Historical and geographic specificity.** German applicants in the early 1990s, DM
  amounts. Findings do not transfer to contemporary lending populations.
- **Small size.** 1,000 rows is a benchmark toy by modern standards; a 200-row test set gives
  wide confidence intervals on every reported metric.
- **No documented consent basis or collection methodology** accompanies the original release.

### Derived dataset — synthetic explanations

`src/training/make_explanations.py` generates plain-English, bank-tone explanation text from
each row's features plus its label, used to LoRA fine-tune the explanation model. This text is
**synthetic and template-derived**, not written by credit analysts, so it constrains the
explainer to generic reasoning. The generator originally used the inverted label convention;
this has been corrected in the repository, though the published adapter was trained before the
fix (see the model card for why inference-time correction is sufficient).

---

## 2. Policy Corpus (RAG)

### Documents

Three publicly available AML/KYC guidance documents, chunked and embedded into a FAISS index.

| Document | Chunks | Publisher | Topic |
|---|---|---|---|
| `basel_aml_cft_2020.pdf` | 289 | Basel Committee on Banking Supervision | Sound management of ML/TF risks |
| `fatf_banking_rba.pdf` | 177 | Financial Action Task Force | Risk-based approach for the banking sector |
| `basel_kyc_cdd.pdf` | 95 | Basel Committee on Banking Supervision | Know-your-customer / customer due diligence |
| **Total** | **561** | | |

### Processing

| Stage | Detail |
|---|---|
| Extraction | `pypdf` |
| Chunking | Fixed-size chunks with per-chunk `id` and `source` metadata |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Index | FAISS `IndexFlatIP`, exact inner-product search over normalised vectors |
| Artefacts | `data/rag/index/` — FAISS index plus `chunks.json` |

### Coverage and limitations

- **Domain is narrow by design:** anti-money-laundering, counter-terrorist financing, KYC and
  customer due diligence. The corpus contains **no** capital-adequacy or Basel III prudential
  material, so questions about Tier 1 capital, capital ratios, or liquidity coverage are
  correctly refused rather than answered.
- **No document versioning.** Regulatory guidance is revised; retrieved passages reflect the
  PDF snapshots committed to the repository, not current rules.
- **No table or figure handling.** Text extraction only; tabular content in the source PDFs
  may be flattened or lost.
- **Chunk boundaries are naive.** Fixed-size chunking can split a requirement across two
  chunks, so a single retrieved passage may be incomplete.
- **Not legal advice.** Outputs are for demonstration only.

### Licensing

All three documents are public guidance published by the Basel Committee and FATF and are
redistributed here for non-commercial educational demonstration. The German Credit dataset is
distributed for research use via the UCI repository. Users should consult the original
publishers for authoritative and current versions.

---

*Companion document: `MODEL_CARD.md`.*
