# 🚦 RiskRadar — IT Incident SLA Breach Risk (Decision Support)

**ML predicts SLA‑breach risk. Dashboard shows the numbers + visuals. Optional LLM summary is *fact‑grounded* (no guessing).**

<br/>

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b?logo=streamlit&logoColor=white)
![ML](https://img.shields.io/badge/ML-scikit--learn-f7931e?logo=scikitlearn&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-92%25-2ea44f)
![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.9684-blue)
![Status](https://img.shields.io/badge/Status-Completed-2ea44f)

---

## 🔎 What this is

RiskRadar is a small **decision-support app** for IT incident teams.

You upload an incident event log (**incident_event_log.csv**) → the pipeline builds **incident-level features** → a trained model outputs an **SLA breach probability** for each incident → the UI shows:

- ✅ **Overview**: total incidents, average risk, counts by risk band  
- 📋 **Incident List**: searchable / sortable risk table  
- 🧾 **Incident Detail**: one incident, its computed features, risk band + recommended action  
- 📊 **Visuals**: distributions, top risky groups/categories, calibration plot, and a heatmap

It's not a "perfect oracle" project. It's more like: *if you're triaging 25k tickets, where should you look first?*

---

## ✅ Why this is useful (in real life)

- ⚡ **Faster triage**: you can filter to High risk tickets quickly
- 🧠 **Consistency**: the risk score is based on the same feature rules every time
- 📈 **Ops insights**: you can see risky assignment groups / categories (patterns show up fast)
- 🧾 **Explainable enough**: explanations are built from computed facts (counts, durations, etc.)

---

## ✨ Features

- 📤 Upload a CSV event log (ServiceNow-like incident event data)
- 🧱 Incident summary builder (event log → incident level table)
- 🤖 SLA breach risk prediction (probability + risk band: Low / Medium / High)
- 🗂️ Sortable / filterable incident list + search by INC number
- 🧾 Incident detail view with:
  - computed features for that incident
  - risk + recommended action
  - optional LLM "short explanation" using only computed facts
- 📊 Dashboard visuals:
  - risk probability distribution
  - risk band counts
  - top risky assignment groups (avg probability)
  - top risky categories (avg probability)
  - calibration curve (how predicted probs match reality)
  - priority × risk band heatmap (quick triage view)
- ⬇️ Download predictions as CSV

---

## 📊 Model Performance (Verified)

Two models were trained and compared. Random Forest was selected as the best performer.

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | 90.0% | **92.0%** |
| AUC-ROC | 0.9589 | **0.9684** |
| F1-Score (SLA Breached) | 0.865 | **0.887** |
| F1-Score (SLA Met) | — | **0.938** |

**Confusion Matrix — Random Forest** (test set: 4,984 incidents):

```
                   Predicted Met    Predicted Breached
Actual Met              3,014               147
Actual Breached           254             1,569
```

The model caught **1,569 SLA breaches** while only missing 254 — an **86% breach recall rate**.

---

## 🚦 Risk Band Logic

The model outputs a probability score `p` (0.0 → 1.0) mapped to an actionable risk band:

| Risk Band | Threshold | Recommended Action |
|---|---|---|
| 🟢 Low | `p < 0.30` | Normal queue. Keep updates clean, avoid unnecessary reassignment. |
| 🟡 Medium | `0.30 ≤ p < 0.60` | Monitor. Check missing details and confirm ownership early. |
| 🔴 High | `p ≥ 0.60` | Escalate now. Assign correctly, reduce reassignment loops, senior review. |

---

## 🗂️ Dataset

- **Source:** `incident_event_log.csv` — UCI ML Repository (Incident management process enriched event log)
- **Raw event logs:** 141,712 rows
- **Aggregated incidents:** 24,918 unique tickets (one row per incident)
- **Train / Test split:** 80% / 20%, stratified by target, random seed 42
- **Target variable:** `sla_breached` — derived from `made_sla` column (0 = Met, 1 = Breached)

---

## 🧩 How the files connect (big picture)

Think of it like a simple pipeline:

1) **data_processing.py**  
   Reads the raw event log CSV and does basic cleanup (dates, missing values, column sanity).

2) **feature_engineering.py**  
   Turns event-level rows into **incident-level features** (counts, reassignments, reopen count, time-based stats, etc.).

3) **model_training.py** + **run_train.py**  
   Trains the model and saves it as `best_model.joblib`.  
   (You run training once. After that, you just load the model.)

4) **decision_logic.py**  
   Converts probability → **risk band** + **recommended action** (simple rules, easy to explain).

5) **llm_explainer.py** (optional)  
   Generates a short explanation text. It only uses computed facts from the incident row.  
   If you don't add an API key, the app still works (it falls back to a non-LLM explanation).

6) **app.py**  
   Streamlit UI that ties everything together.

---

## 🗂️ Project structure

```text
riskradar-it-incident-sla-risk/
├─ app.py
├─ data_processing.py
├─ feature_engineering.py
├─ model_training.py
├─ run_train.py
├─ decision_logic.py
├─ llm_explainer.py
├─ requirements.txt
├─ incident_event_log.csv              # (optional) local copy (large)
├─ best_model.joblib                   # trained model (generated after training)
└─ RiskRadar_Report.ipynb              # your notebook report (optional)
```

---

## 🛠️ Setup

### 1) Create a venv (recommended)

```bash
python -m venv .venv
```

**Windows (PowerShell):**
```bash
.venv\Scripts\Activate.ps1
```

**Mac/Linux:**
```bash
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 How to run

### Option A: Run the app (normal)

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

### Option B: Train the model (only if you want to retrain)

```bash
python run_train.py
```

After training, it will create/update `best_model.joblib`.

---

## 🎥 Demo video 

https://github.com/user-attachments/assets/b49adc74-d1b8-49e3-b63d-4d563a4166a0

---

## 📊 About the visuals (what each one means)

- **Risk probability distribution**: shows how the model spreads predictions (lots of low, some high, etc.).  
- **Risk band counts**: quick count of Low/Medium/High based on your thresholds.  
- **Top risky assignment groups / categories**: average predicted risk by group/category (helps spot patterns).  
- **Calibration curve**: checks if predicted probabilities are realistic (e.g., "0.8 means ~80% breach").  
- **Priority × Risk heatmap**: where risk is concentrated across priority labels (fast triage view).

---

## 🧠 Optional LLM explanations (how to enable)

By default, the app shows a **non‑LLM explanation** (template-based) so it works anywhere.

If you want the LLM version:
1) pick a provider (OpenAI / Azure OpenAI / etc.)
2) set an env var (example below)

**Example (OpenAI):**
```bash
setx OPENAI_API_KEY "your_key_here"
```

Then re-open your terminal and run:
```bash
streamlit run app.py
```

> Note: the LLM is only used to *summarise computed facts* (counts, durations, band, etc.).  
> It should not invent incident details that aren't in the CSV.

---

## 📌 Dataset source & credits

This project uses the **Incident management process enriched event log** dataset from the UCI Machine Learning Repository.

**Citation (APA):**  
Amaral, C., Fantinato, M., & Peres, S. (2018). *Incident management process enriched event log* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C57S4H

License: CC BY 4.0 (as listed on the dataset page).

---

## 🧪 Notes / limitations (keeping it honest)

- This is a **prototype**. Real incident systems need access control, monitoring, audit logs, and careful evaluation.
- The dataset comes from a single organisation, generalisation to other environments would need retraining.
- Labels/features depend on what the dataset provides. If a company tracks different fields, you'd adapt feature engineering.
- If you retrain, results can change slightly (random splits / model settings).

---

## 🙋 Author

**Abinash Prasana Selvanathan**  

---

### ⭐ If you like it
If you found it useful, feel free to star the repo — it helps.
