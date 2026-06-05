# PRISM: A Personalized News Recommendation System for Mitigating Political Echo Chambers

Supplementary code for the paper submitted to CIKM 2026.
This repository implements the full pipeline: political bias classification with Integrated Gradients explanations (IG_khan) and perspective-balanced news recommendation via HDBSCAN clustering (news_clustering), designed to expose users to diverse political viewpoints and reduce echo chamber effects.

---

## Project Structure

```
Unified/
├── data/                    # Datasets (not included — see below)
├── models/
│   └── khan/                # KHAN checkpoint + vocab (not included)
├── IG_khan/                 # Bias classification + Integrated Gradients XAI
├── news_clustering/         # HDBSCAN clustering + recommendation
├── common/                  # Shared data loader
├── scripts/                 # Utility scripts
└── outputs/                 # Pipeline outputs (generated at runtime)
    ├── IG_khan/
    ├── news_clustering/
    └── recommendations/
```

---

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

GPU (recommended for IG analysis):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Data

Download from [KHAN official Google Drive](https://drive.google.com/drive/u/2/folders/1ksV0PUncXyBnEHGPB4H4mae2ybXX3Ch0) and place in `data/`:

| File | Description |
|---|---|
| `AllSides-L_test.csv` | ~71.9k articles (10-fold, 1st fold test split) |
| `AllSides-L.csv` | Full dataset (~719k) — only needed for `--full` |

Pre-trained KG embeddings → `IG_khan/pre-trained/`:
[KHAN Drive](https://drive.google.com/drive/u/2/folders/14EgeI1RdSTccETqRgDd36writP6lUu1R)

---

## Models

Train using the official repo and place checkpoints in `models/khan/`:

| Model | Repo | Files needed |
|---|---|---|
| KHAN | [yy-ko/khan-www23](https://github.com/yy-ko/khan-www23) | `allsides-l_best.pt`, `vocab_train.pkl` |

---

## Running the Pipeline

### Step 1 — Compute predicted labels (KHAN inference on full test set)
```bash
python scripts/add_predicted_labels.py
```

### Step 2 — IG keyword extraction
```bash
cd IG_khan

# Sample (default: 5 articles)
python run_ig.py

# Full test set
python run_ig.py --all
```

### Step 3 — News clustering
```bash
cd ../news_clustering
python main.py
```

### Step 4 — Alternative article recommendation
```bash
python recommend.py
```

### Step 5 — Similarity statistics (optional)
```bash
python compute_similarity_stats.py
```

**Output locations:**
- IG results: `outputs/IG_khan/article_{id}.json`
- Clustering: `outputs/news_clustering/all-MiniLM-L6-v2/`
- Recommendations: `outputs/recommendations/recommendations.json`

---

## Citation

```bibtex
@inproceedings{}
```
