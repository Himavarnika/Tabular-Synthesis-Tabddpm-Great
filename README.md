# 📊 Synthetic Tabular Data Generation using GReaT & TabDDPM

## 🚀 Overview
This project focuses on generating synthetic tabular data for a bank customer churn dataset using two approaches:

- GReaT (LLM-based) — converts tabular data into text and uses GPT-2  
- TabDDPM (Diffusion-based) — directly models tabular data distributions  

The goal is to generate data that is:
- Realistic  
- Useful for machine learning  
- Privacy-preserving  

---

## 🎯 Problem Statement
Banking datasets are:
- Sensitive (privacy constraints)  
- Limited (restricted access)  
- Imbalanced (few churn cases)  

This project generates synthetic data to safely replace or augment real data.

---

## 📂 Dataset
- Botswana Bank Customer Churn Dataset  
- ~1,15,000 rows × 25 columns  

Features include:
- Numerical: Income, Balance, Credit Score  
- Categorical: Gender, Occupation, Segment  
- Target: Churn Flag (0/1)  

---

## ⚙️ Methodology

### Phase 1: Data Preparation
- Dropped identifiers and leakage columns  
- Handled missing values (median/mode)  
- Encoded categorical features  
- Scaled numerical features  
- Train/Test split (test kept real-only)  

---

### Phase 2: Model Training

#### GReaT (LLM-Based)
- Convert rows → text sentences  
- Fine-tune distilgpt2  
- Generate data token-by-token  
- Parse text → structured CSV  

#### TabDDPM (Diffusion-Based)
- Train diffusion model on tabular data  
- Add noise → learn reverse denoising  
- Generate structured synthetic rows  

---

### Phase 3: Evaluation

Metrics used:
- JSD (Jensen-Shannon Distance) → distribution similarity  
- Correlation L2 → feature relationships  
- Density → realism  
- Coverage → diversity  

---

## 📊 Results

| Metric | TabDDPM | GReaT |
|-------|--------|-------|
| Mean JSD ↓ | 0.000257 | 0.2167 |
| Correlation L2 ↓ | 0.839 | 0.849 |
| Density ↑ | 0.786 | 0.711 |
| Coverage ↑ | 0.917 | 0.879 |

TabDDPM performs better in distribution matching, realism, and diversity.

---

## 🔍 Key Insights
- GReaT captures feature relationships well using attention  
- But struggles with structure and distribution accuracy  
- TabDDPM produces more stable and realistic tabular data  

---

## ⚠️ Challenges

GReaT Issues:
- Duplicate / invalid values  
- Unstructured text output  
- Slow CPU generation  

Solutions:
- Custom parsing and validation  
- Data cleaning and schema enforcement  
- Controlled generation parameters  

---

## 🧪 Tech Stack
- Python  
- PyTorch  
- HuggingFace Transformers (distilgpt2)  
- Scikit-learn  
- Pandas, NumPy  
- Matplotlib, Seaborn  

---

## 📁 Project Structure

    project-root/
    ├── data/
    ├── preprocessing/
    ├── great_model/
    ├── tabddpm_model/
    ├── evaluation/
    ├── outputs/
    │   ├── synthetic_great.csv
    │   ├── synthetic_tabddpm.csv
    │   ├── plots/
    │   └── reports/
    └── README.md

---

## 🔮 Future Work
- Train on full dataset using GPU  
- Compare with CTGAN, TVAE  
- Hyperparameter tuning  
- Conditional data generation  
- Deploy as API  

---

## 📌 Conclusion
- GReaT is strong for learning feature relationships  
- TabDDPM is better for statistical accuracy and stability  

TabDDPM is more suitable for high-quality synthetic tabular data generation.

---

## 📚 References
- Borisov et al. (2022) — Language Models are Realistic Tabular Data Generators  
- TabDDPM Paper — Modelling Tabular Data with Diffusion Models  
- HuggingFace Transformers  

---

## 👩‍💻 Author
Your Name  
https://github.com/your-username
