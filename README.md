# Mini Project 9: Content Moderation with Transformers

**COMP 9130 — Applied Artificial Intelligence**  
**Team:** Group 8 - Binger Yu & Savina Cai    
**Due:** Week 11

---

## Problem Description & Motivation

**SafeSpace AI** is a startup building automated content moderation tools for social media platforms. Manual moderation is expensive, slow, and psychologically taxing for human reviewers.

The system classifies social media posts into three categories:

| Class | Label | Description |
|---|---|---|
| 0 | Hate Speech | Attacks individuals/groups based on protected characteristics |
| 1 | Offensive Language | Rude or vulgar, but not group-targeted |
| 2 | Neither | Acceptable content requiring no moderation action |

The critical challenge is **distinguishing hate speech from offensive language** — a distinction with legal implications. Platforms that fail to remove hate speech face regulatory penalties; over-censoring offensive (but legal) speech damages user trust.

---

## Dataset

**Twitter Hate Speech and Offensive Language Dataset**  
Davidson et al. (2017). *Automated Hate Speech Detection and the Problem of Offensive Language.* ICWSM.

- **Source:** https://github.com/t-davidson/hate-speech-and-offensive-language
- **Direct CSV:** `labeled_data.csv`
- ~24,800 tweets annotated by CrowdFlower workers (min 3 raters each)
- **Severely imbalanced:** ~5% hate speech | ~77% offensive | ~18% neither
- Text includes slang, abbreviations, URLs, @mentions, hashtags

⚠️ This dataset contains offensive and hateful language. It is used professionally for harm-reduction research.

---

## Setup & How to Run

### Option 1: Google Colab (recommended)

Upload the notebooks to Colab and run cells in order. The dataset is automatically downloaded in `01_exploration.ipynb`.

### Option 2: Local

```bash
# Clone or download the repository
git clone https://github.com/bing-er/mini-project-9.git
cd mini-project-9

# Install dependencies
pip install -r requirements.txt

# Run notebooks in order
jupyter notebook notebooks/01_exploration_colab.ipynb
jupyter notebook notebooks/02_baseline_colab.ipynb
jupyter notebook notebooks/03_transformer.ipynb
```

The dataset is auto-downloaded in `01_exploration.ipynb`. No manual download required.

---

## Repository Structure

```
mini-project-9/

├── data/
│   ├── labeled_data.csv        # Raw dataset (auto-downloaded)
│   ├── train.csv               # 70% split (stratified)
│   ├── val.csv                 # 15% split (stratified)
│   ├── test.csv                # 15% split (stratified)
│   └── download_dataset.txt    # Instructions for downloading the dataset
├── figures/
│   ├── baseline/
│   └── DistilBERT/
├── notebooks/
│   ├── 01_exploration.ipynb    # Data exploration & preprocessing (Binger)
│   ├── 02_baseline.ipynb       # TF-IDF + classifier baseline (Binger)
│   └── 03_transformer.ipynb    # DistilBERT fine-tuning & analysis (Savina)
├── src/
│   └── __init__.py 
├── .gitignore
├── README.md
├── requirements.txt
```

---

## Results Summary

| Metric | TF-IDF + LR Baseline | DistilBERT Fine-tuned | Improvement 
|---|---|---|-------------|
| Overall Accuracy | 0.8701 | 0.9080 | +4.35%      |
| Macro F1 | 0.7301 | 0.7828 | +7.22%      |
| F1 — Hate Speech (Class 0) | 0.4137 | 0.5115 | +23.65%     |
| F1 — Offensive (Class 1) | 0.9195 | 0.9448 | +2.75%      |
| F1 — Neither (Class 2) | 0.8571 | 0.8920 | +4.07%      |

**Key finding:**
1. Transformer Superiority: The DistilBERT model consistently outperformed the TF-IDF baseline across all metrics. The most significant improvement was observed in the minority class (Hate Speech), where the F1 score increased by 23.65% relative (from 0.41 to 0.51). This demonstrates the transformer's ability to capture semantic context and subtle differences between hate speech and general offensive language, which bag-of-words approaches fail to recognize.

2. Production Workflow Efficiency: By implementing a confidence-based, four-tier workflow (Auto-Remove, Auto-Flag, Auto-Approve, and Human Review), we achieved a 97.0% automation rate. At a scale of 100,000 posts per day, the system reduces the required human review volume to just 3.0% (3,039 posts), requiring only 1.9 full-time equivalent (FTE) reviewers compared to 62 FTEs for manual review.



---

## Team Contributions

| Task                                       | Owner      |
|--------------------------------------------|------------|
| Part 1: Data Exploration & Preprocessing   | Binger Yu  |
| Part 2: TF-IDF Baseline (LR, SVM, RF)      | Binger Yu  |
| GitHub Repository, README & Report Writing | Binger Yu  |
| Part 3: DistilBERT Fine-tuning             | Savina Cai |
| Part 4: Comparative & Error Analysis       | Savina Cai |
| Part 5: Production Workflow Design         | Savina Cai |

---

## References

Davidson, T., Warmsley, D., Macy, M., \& Weber, I. (2017). Automated hate speech detection and the problem of offensive language. \textit{Proceedings of the 11th International AAAI Conference on Web and Social Media (ICWSM)}, 512--515. https://github.com/t-davidson/hate-speech-and-offensive-language