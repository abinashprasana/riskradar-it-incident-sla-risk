# 🚦 RiskRadar — IT Incident SLA Breach Risk Predictor

<p align="center">
  <b>An ML-based decision support system for IT incident triage</b><br/>
  Predicts SLA breach risk using historical incident event logs
</p>

---

## 🧠 Overview

RiskRadar is built around a very practical IT operations problem:

> *When hundreds or thousands of incidents are open, which ones are actually risky and need attention now?*

Most ITSM tools rely heavily on static priority labels. In reality, incidents evolve over time — reassignments, reopenings, long inactivity gaps, and category-specific patterns all affect whether an SLA breach is likely.

This project takes **real incident event history**, learns from it, and produces:
- A **probability-based SLA breach risk**
- Clear **risk bands (Low / Medium / High)**
- Simple **recommended actions**
- Visuals that help teams understand *where* and *why* risk is building up

This is not just a model — it’s a small end-to-end system.

---

## 🎯 Key Features

- 📊 SLA breach **probability prediction** (not just labels)
- 🧩 Feature engineering from event-level data
- 🎚️ Probability calibration to align predictions with reality
- 🚦 Risk banding (Low / Medium / High)
- 🧠 Human-readable explanations (facts-based)
- 📈 Interactive dashboard (Streamlit)
- 📥 Downloadable scored dataset for further analysis

---

## 📚 Dataset

**Incident management process enriched event log**  
Source: **UCI Machine Learning Repository**

This dataset comes from a real ServiceNow incident management system and contains anonymized, enriched event logs describing how incidents evolve over time.

**Citation (APA):**

> Amaral, C., Fantinato, M., & Peres, S. (2018). *Incident management process enriched event log*.  
> UCI Machine Learning Repository. https://doi.org/10.24432/C5754H

---

## 🏗️ Project Structure

```
riskradar-it-incident-sla-risk/
│
├── app.py                  🖥️ Streamlit dashboard (main entry point)
├── data_processing.py      🧹 Data loading, cleaning, aggregation
├── feature_engineering.py  🧠 Feature creation for ML model
├── model_training.py       🤖 Model training + calibration
├── run_train.py            ▶️ Script to train and save model
├── decision_logic.py       🚦 Risk bands + recommended actions
├── llm_explainer.py        💬 Optional explanation layer (facts only)
├── best_model.joblib       📦 Trained model artifact
├── incident_event_log.csv  📄 Input dataset
├── requirements.txt        📌 Dependencies
└── RiskRadar_Report.ipynb  📘 Detailed project explanation (notebook)
```

Each module has a clear responsibility so the code stays readable and easy to reason about.

---

## 🔗 How the pieces fit together

### 1️⃣ `data_processing.py`
- Reads the raw event log
- Handles missing values and timestamps
- Aggregates event-level data into **incident-level summaries**

👉 Output: one clean row per incident

---

### 2️⃣ `feature_engineering.py`
- Converts summaries into numeric features
- Examples:
  - Total number of events
  - Reassignment count
  - Reopen count
  - Average gaps between events

👉 Output: model-ready feature matrix

---

### 3️⃣ `model_training.py`
- Trains a classification model
- Applies **probability calibration**
- Saves the trained model to disk

👉 Output: `best_model.joblib`

---

### 4️⃣ `decision_logic.py`
- Maps probabilities to **risk bands**
- Attaches **recommended actions**
- Keeps business logic separate from ML code

👉 Output: interpretable risk decisions

---

### 5️⃣ `llm_explainer.py`
- Generates short explanations
- Uses only computed facts (no guessing)
- Optional layer — model works without it

👉 Output: human-friendly explanations

---

### 6️⃣ `app.py`
- Loads the trained model and data
- Runs predictions
- Displays dashboard, tables, filters, and downloads

👉 This is what the user actually interacts with

---

## 📊 Dashboard & Visuals (what they show)

Each visual answers a specific operational question:

- 📈 **Risk probability distribution**  
  Shows how incidents are spread across low → high risk.

- 🚦 **Risk band counts**  
  Quick view of how many incidents need attention.

- 👥 **Top risky assignment groups**  
  Highlights teams where SLA breaches are more common.

- 🗂️ **Top risky categories**  
  Identifies problem areas in the IT landscape.

- 🔥 **Priority × Risk band heatmap**  
  Shows where risk concentrates by priority level — useful for quick triage.

- 🎯 **Calibration curve**  
  Checks whether predicted probabilities align with actual outcomes.

Each chart is explained inside the dashboard in simple language so a new user isn’t lost.

---

## ▶️ Demo Video

📹 **Demo walkthrough (to be added)**

Planned demo will show:
- Uploading the dataset
- Exploring the dashboard
- Understanding risk scores
- Downloading predictions

*(Link will be added once recorded)*

---

## 🚀 How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train the model
```bash
python run_train.py
```

### Launch the dashboard
```bash
streamlit run app.py
```

---

## 📤 Outputs

- Interactive dashboard in browser
- Sortable & filterable incident list
- Incident-level risk details
- Downloadable CSV with predictions

---

## 🧪 Project Level

**Intermediate → Advanced**

This project demonstrates:
- Applied machine learning
- Feature engineering from real event logs
- Probability calibration
- Clear separation of concerns
- Practical decision-support design

---

## 📝 Final Notes

RiskRadar is built to be:
- Understandable
- Explainable
- Useful in real operations

Every score, chart, and recommendation can be traced back to actual incident behavior — no black boxes, no magic.

