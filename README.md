<div align="center">

# 🛰️ RiskRadar — IT Incident SLA Breach Risk (Decision Support)
### ML Risk Scoring • Visual Dashboard • Human-Readable Explanations • Exportable Results

<p>
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/Data-pandas-150458?logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen" />
</p>

</div>

---

## 👀 What this is

**RiskRadar** is a small decision-support app for **IT incident triage**.

You upload an incident event log → the app **summarises each incident** into useful features → a trained ML model predicts the **probability of SLA breach** → the app groups incidents into **risk bands** (High/Medium/Low), gives a **recommended next action**, and shows a **dashboard with visuals** so it’s easy to spot patterns.

It’s meant to be practical: something you can demo in an interview and also explain clearly.

---

## ✅ Why this project is useful

In real IT operations, teams don’t just need a prediction — they need:
- **a risk score** they can trust (probability, not just labels)
- **a reason / explanation** in plain language (based on computed signals)
- **visibility across the queue** (dashboard + filters)
- **a way to export results** (CSV) for reporting / handover

That’s basically what RiskRadar is doing.

---

## ✨ Features

- 📁 Upload **incident_event_log.csv** (event-level log)
- 🧠 Builds **incident-level summaries** (counts, stats, churn signals)
- 🎯 Predicts **SLA breach probability** using a saved model (`best_model.joblib`)
- 🧭 Converts probability → **risk band** (High / Medium / Low)
- 🧾 Generates short **human-readable explanations** (grounded in computed facts)
- 📊 Dashboard visuals:
  - Risk probability distribution
  - Risk band counts
  - Top risky assignment groups (avg probability)
  - Top risky categories (avg probability)
  - 📌 Priority × Risk band heatmap (triage view)
  - 📈 Calibration curve (to sanity-check probability behavior)
- 🔎 Incident list (sortable / searchable)
- 🧩 Incident detail view (for a single ticket)
- ⬇️ Download scored dataset as CSV

---

## 🧠 How the app “thinks” (simple version)

1. **Event logs → Incident summary**  
   Many rows per incident → summarised into 1 row per incident.

2. **Summary features → ML risk score**  
   Model predicts probability: `P(SLA_Breach = 1)`.

3. **Probability → Risk band + action**  
   Example:
   - High risk → escalate, reduce reassignment loops, senior review
   - Medium → monitor closely, ensure proper updates
   - Low → normal queue, avoid unnecessary churn

4. **Dashboard → patterns**  
   Helps answer:
   - Which assignment groups are repeatedly risky?
   - Which categories tend to breach?
   - Is priority correlated with high-risk? (heatmap)
   - Are predicted probabilities roughly calibrated? (calibration plot)

---

## 🧩 Workflow Highlights

<details>
  <summary><b>📦 Data processing (event log → clean types)</b></summary>

- Handles missing values safely  
- Parses time columns (best-effort)  
- Keeps only fields needed for summarisation + model scoring  
</details>

<details>
  <summary><b>🧱 Feature engineering (incident-level signals)</b></summary>

Examples of signals created (depends on dataset columns available):
- total events in an incident
- max / mean change counts (status, assignment, etc.)
- reassignment / reopen churn
- priority / category / assignment group encoded safely

</details>

<details>
  <summary><b>🎯 Model scoring</b></summary>

- Loads `best_model.joblib`
- Predicts **probability** (not just 0/1)
- Applies risk band thresholds
</details>

<details>
  <summary><b>🧾 Explanation generation</b></summary>

Explanations are generated using **computed values** only:
- probability, risk band
- churn counts / event counts
- priority/category/group patterns (when available)

No “magic model reasoning” claims — it stays grounded.
</details>

---

## 🗂️ Project Structure

```text
riskradar-it-incident-sla-risk/
├── app.py
├── data_processing.py
├── feature_engineering.py
├── model_training.py
├── decision_logic.py
├── llm_explainer.py
├── run_train.py
├── requirements.txt
├── RiskRadar_Report.ipynb
└── (your local files)
    ├── incident_event_log.csv          # optional locally (large file)
    └── best_model.joblib               # generated after training
```

---

## 🧾 File-by-file (what each file does)

### `app.py` (main entry)
- Streamlit UI (Dashboard / Incident List / Incident Detail)
- Calls processing + feature engineering + scoring
- Renders charts and tables
- Exports scored results as CSV

### `data_processing.py`
- Loads CSV safely
- Handles missing values + type cleanup
- Prepares a clean dataframe for feature engineering

### `feature_engineering.py`
- Converts event-level data into incident-level summary features
- Produces the single-row-per-incident table used for scoring

### `model_training.py`
- Training pipeline
- Splits data, trains ML model, evaluates
- Saves best model as `best_model.joblib`

### `run_train.py`
- Small runner script to train from terminal (quick and clean)

### `decision_logic.py`
- Converts probability into:
  - `risk_band`
  - `recommended_action`
- Keeps decision rules in one place

### `llm_explainer.py`
- Generates short explanation text
- Stays grounded in computed facts (probability + summary signals)

### `RiskRadar_Report.ipynb`
- Notebook version of the project write-up / walkthrough
- Useful for explaining approach + results

---

## 🚀 How to Run (IDE / Terminal)

### 1) Create venv (recommended)
```bash
python -m venv .venv
```

Activate:

**Windows (PowerShell)**
```bash
.\.venv\Scripts\Activate.ps1
```

**Mac/Linux**
```bash
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Run the app
```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

---

## 🏋️ Train a model (optional)

If you don’t have a model yet, train one:

```bash
python run_train.py --data incident_event_log.csv --out best_model.joblib
```

After that, run the Streamlit app and set **Model path** to `best_model.joblib`.

> If you already have `best_model.joblib`, you can skip training and just run the app.

---

## 📊 Visuals (what you’re seeing)

Inside the **Dashboard**:
- **Overview**: total incidents, average predicted breach probability, counts by risk band
- **Distribution**: shows whether the queue is mostly low-risk or skewing high-risk
- **Risk band counts**: quick queue composition
- **Top risky groups/categories**: highlights where risk concentrates
- **Priority × Risk heatmap**: shows where high risk clusters by priority
- **Calibration curve**: sanity check for probability behavior (not perfect, but useful)

---

## 📌 Dataset Source & Citation

This project uses the UCI ML Repository dataset:

**Incident management process enriched event log**  
Creators: Claudio Amaral, Marcelo Fantinato, Sarajane Peres

**APA citation (from UCI)**:
> Amaral, C., Fantinato, M., & Peres, S. (2018). *Incident management process enriched event log* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C57S4H

License noted on the UCI page: **CC BY 4.0** (please keep attribution if you redistribute).

---

## ⚠️ Notes (practical)

- The uploaded dataset can be large — so the app is designed to work from a file upload.
- The model is trained on engineered features from this dataset; if you swap datasets, you’ll likely need retraining.
- Explanations are “safe”: they describe computed patterns and suggested next steps (not fake model reasoning).

---

## 🧠 Skills shown in this project

- Data preprocessing (real log data)
- Feature engineering (event → incident summarisation)
- ML model training + probability scoring
- Decision logic layer (risk band + recommended action)
- Streamlit dashboard development
- Basic model validation visuals (calibration curve)
- Clean project structure + reproducible runs

---

## ✍️ Author

**Abinash Prasana (Abby)**  
GitHub: `abinashprasana`
