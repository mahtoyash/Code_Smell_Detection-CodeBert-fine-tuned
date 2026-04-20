# 🔍 CodeSmellDetector

A fine-tuned [`microsoft/codebert-base`](https://huggingface.co/microsoft/codebert-base) model for detecting **5 classes of Python code smells**, trained on a hybrid dataset of real open-source code and synthetic hard examples.

---

## 📋 Detected Smell Classes

| Label | Class | Detection Rule |
|-------|-------|----------------|
| `0` | **Long Method** | Function > 20 lines *or* cyclomatic complexity > 10 |
| `1` | **Large Parameter List** | Function with > 5 parameters |
| `2` | **God Class** | Class with ≥ 7 methods *and* ≥ 10 attributes, or ≥ 12 methods alone |
| `3` | **Feature Envy** | Method accesses external objects more than `self` |
| `4` | **Clean Code** | No smells detected |

---

## 🏗️ Architecture & Pipeline

```
Real Repos (43 OSS codebases)
        │
        ▼
  AST Snippet Extraction
  (functions + classes)
        │
        ▼
  Rule-Based Labeling          God Class Mining
  (radon + AST heuristics)  +  (10 targeted repos)
        │                              │
        └──────────┬───────────────────┘
                   ▼
         Synthetic Data Generation
         (60,000 hard examples, 5 classes)
                   │
                   ▼
         Balanced Dataset (12,000/class)
                   │
                   ▼
      Fine-tune CodeBERT (5 epochs)
                   │
                   ▼
      Flask API + Heuristic Fallback
```

---

## 📦 Dataset Construction

### Real Data Sources

**43 production Python repositories** including:

- Web frameworks: Django, Flask, FastAPI, Sanic, Starlette
- Data/ML: pandas, scikit-learn, Keras, matplotlib
- DevOps: Ansible, Airflow, Celery, Luigi
- Tooling: pytest, pip, black, click, SQLAlchemy

**God class specialist repos** (Cell 4b miner):

- Odoo, Home Assistant, Wagtail, OpenStack Nova, Saleor, and more — codebases known to contain real-world god classes.

### Synthetic Data (Cell 5)

60,000 hard examples across 5 generators — each designed to produce *difficult* cases that challenge the classifier:

| Generator | Count | Patterns |
|-----------|-------|----------|
| Long Method | 8,000 + edge cases | Multi-step pipelines, nested conditionals, long loops, exception chains, state machines |
| Large Param List | 10,000 + edge cases | Cryptic single-letter, typed with defaults, API handler style, DB connection style |
| God Class | 11,400 + edge cases | Blob, Swiss Army, Legacy Grower, Utility Dump, Deep Coordinator, Inheriting God Class |
| Feature Envy | 9,000 + edge cases | Field summation, validation, formatting, price computation, bulk field updates |
| Clean Code (hard negatives) | 5,600 | Django models, dataclasses, ABCs, adapters, self-heavy methods, boundary functions |

Final dataset is balanced to **12,000 examples per class (60,000 total)**, split 80/10/10 (train/val/test).

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch transformers datasets accelerate radon pandas matplotlib scikit-learn tqdm seaborn flask
```

### 2. Run the Notebook

Run cells in order:

| Cell | Purpose |
|------|---------|
| 1 | Environment setup & GPU check |
| 2 | Clone 43 OSS repos |
| 3 | AST snippet extraction |
| 4 | Rule-based labeling |
| 4b | Mine real god classes from 10 targeted repos |
| 5 | Build balanced dataset (synthetic + real) |
| 6 | Build HuggingFace `DatasetDict` |
| 7 | Tokenize with CodeBERT tokenizer |
| 8 | Load model |
| 9 | Configure training |
| 10 | Train (5 epochs) |
| 11 | Save model |
| 12 | Evaluate (confusion matrix) |
| 12b | True evaluation — real code only |
| 13 | Inference functions |
| 14 | Demo |
| 15b | Start Flask server |
| 15c | Smoke test all 5 classes |

### 3. Start the API Server

```python
# Cell 15b starts a Flask server automatically on port 5000
# Or run standalone:
python server.py
```

### 4. Predict

```python
import requests

code = """
def process(a, b, c, d, e, f, g):
    return a + b
"""

response = requests.post("http://localhost:5000/predict", json={"code": code})
print(response.json())
```

```json
{
  "smell": "large_param_list",
  "confidence": 0.9821,
  "heuristic_override": false,
  "raw_model_label": "large_param_list",
  "all_scores": {
    "long_method": 0.0041,
    "large_param_list": 0.9821,
    "god_class": 0.0023,
    "feature_envy": 0.0078,
    "clean_code": 0.0037
  }
}
```

---

## 🌐 API Reference

### `POST /predict`

Classify a Python code snippet.

**Request body:**
```json
{ "code": "def foo(a, b): return a + b" }
```

**Response:**
```json
{
  "smell":              "clean_code",
  "confidence":         0.9712,
  "heuristic_override": false,
  "raw_model_label":    "clean_code",
  "all_scores": { ... }
}
```

| Field | Description |
|-------|-------------|
| `smell` | Final predicted class (after heuristic override if applicable) |
| `confidence` | Model confidence for the predicted class |
| `heuristic_override` | `true` if god class heuristic overrode the model |
| `raw_model_label` | Model's original prediction before heuristic |
| `all_scores` | Softmax probabilities for all 5 classes |

### `GET /health`

```json
{ "status": "ok", "device": "cuda", "classes": ["long_method", "large_param_list", "god_class", "feature_envy", "clean_code"] }
```

---

## ⚙️ Heuristic Fallback

A lightweight AST-based fallback catches god classes that the model predicts as `clean_code`:

```
If model predicts "clean_code" AND:
  - ≥ 1 class defined in snippet
  - ≥ 5 methods total
  - ≥ 15 non-empty lines
→ Override to "god_class"
```

This compensates for the model's tendency to under-predict god classes at the boundary.

---

## 🏋️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `microsoft/codebert-base` |
| Epochs | 5 |
| Batch size | 8 (effective 16 with gradient accumulation) |
| Gradient accumulation steps | 2 |
| Learning rate | 2e-5 |
| Warmup steps | 500 |
| Weight decay | 0.01 |
| Max token length | 512 |
| Best model metric | Macro F1 |
| Mixed precision | fp16 (if CUDA available) |

---

## 📊 Evaluation

Two evaluation modes are provided:

**Standard evaluation** (Cell 12) — full test set including synthetic examples.

**True evaluation** (Cell 12b) — **real code only**, filtering out synthetic snippets by naming patterns. This is the honest number; synthetic-only accuracy is inflated because the model partially memorizes generator patterns.

```
Reported (real + synthetic) : ~97%  ← inflated
Synthetic-only              : ~99%  ← model memorized patterns  
TRUE (real code only)       : ~82%  ← what actually matters
```

---

## 📁 Project Structure

```
.
├── CodeSmellDetector.ipynb     # Main notebook (all 15 cells)
├── cloned_repos/               # OSS repos cloned in Cell 2
├── repos_godclass/             # God class specialist repos (Cell 4b)
├── best_codesmell_model/       # Best checkpoint saved during training
│   ├── config.json
│   ├── label_config.json       # Class names & id2label mapping
│   ├── pytorch_model.bin
│   └── tokenizer files
├── codesmell_model_final/      # Final saved model (Cell 11)
├── confusion_matrix.png        # Full test set confusion matrix
└── confusion_matrix_real_only.png  # Real-code-only confusion matrix
```

---

## 🔧 Thresholds

All detection thresholds are defined in Cell 4 and can be adjusted:

```python
LONG_METHOD_LINES       = 20   # Lines threshold for long method
LONG_METHOD_CC          = 10   # Cyclomatic complexity threshold
GOD_CLASS_METHODS       = 7    # Min methods for god class
GOD_CLASS_ATTRIBUTES    = 10   # Min attributes for god class
LARGE_PARAM_COUNT       = 5    # Min parameters for large param list
FEATURE_ENVY_THRESHOLD  = 3    # Min external references for feature envy
```

---

## 📝 Notes

- Test files are excluded from snippet extraction (filenames containing `"test"`).
- Snippets shorter than 50 characters or longer than 8,000 characters are skipped.
- God class mining skips migration, generated, test, conftest, setup.py, and alembic files.
- The dataset is deduplicated by hashing the first 200 characters of each snippet.

---

## 📄 License

MIT
