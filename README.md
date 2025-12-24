# 🚦 RiskRadar — IT Incident SLA Breach Risk Predictor

RiskRadar is a small but serious attempt to answer a very practical question that comes up in IT operations all the time:

**“Which incidents are likely to breach SLA, and where should we focus first?”**

Instead of relying only on static priority labels, this project looks at historical incident activity and predicts **SLA breach risk as a probability**, then presents the results in a clean, interactive dashboard that can actually be used by an operations or support team.

This is not a toy ML notebook. It’s a complete mini system — data processing, model training, risk scoring, explanations, and a UI to explore everything.

---

## 🎯 What this project does

- Predicts **SLA breach probability** for IT incidents  
- Groups incidents into **Low / Medium / High risk bands**
- Shows **where risk concentrates** (by priority, category, assignment group)
- Helps with **triage and decision‑making**, not just prediction
- Provides **human‑readable explanations** instead of black‑box scores

---

## 🧠 Why I built this

In many ITSM tools, priority is fixed early and rarely revisited. But in reality:
- Incidents change over time  
- Reassignments, reopenings, and delays increase risk  
- Some groups and categories consistently struggle more than others  

This project tries to capture that behavior using real historical data and turn it into something **actionable**.

---

## 📊 Dataset

**Incident management process enriched event log**  
Source: UCI Machine Learning Repository

This dataset contains real (anonymized) incident event histories extracted from a ServiceNow system and enriched with relational data.

Citation (APA):
> Amaral, C., Fantinato, M., & Peres, S. (2018). *Incident management process enriched event log*. UCI Machine Learning Repository. https://doi.org/10.24432/C5754H

---

## 🏗️ Project structure

```
riskradar-it-incident-sla-risk/
│
├── app.py                  # Streamlit application (dashboard + UI)
├── data_processing.py      # Data loading, cleaning, aggregation
├── feature_engineering.py  # Feature creation used by the model
├── model_training.py       # Model training and calibration
├── run_train.py            # Script to train and save the model
├── decision_logic.py       # Risk bands + recommended actions
├── llm_explainer.py        # Optional explanation layer (facts-based)
├── best_model.joblib       # Trained model artifact
├── incident_event_log.csv  # Dataset (not committed if large)
├── requirements.txt        # Python dependencies
└── RiskRadar_Report.ipynb  # Detailed explanation notebook
```

Each file has a clear responsibility. Nothing is mixed randomly.

---

## 🔗 How everything connects (high level)

1. **data_processing.py**
   - Loads the raw event log
   - Aggregates events at incident level
   - Produces a clean summary table

2. **feature_engineering.py**
   - Converts summaries into model‑ready features
   - Counts events, reassignments, reopenings, etc.

3. **model_training.py**
   - Trains a classification model
   - Applies probability calibration
   - Saves the final model

4. **decision_logic.py**
   - Converts probabilities into risk bands
   - Generates recommended actions

5. **llm_explainer.py**
   - Turns numeric signals into short explanations
   - Uses only computed facts (no hallucination)

6. **app.py**
   - Loads everything
   - Renders the dashboard, visuals, filters, and downloads

---

## 📈 Dashboard visuals (what they actually mean)

The dashboard is not just “pretty charts”. Each visual answers a specific question:

- **Risk probability distribution**  
  Shows how incidents spread across low → high risk. Helps check model behavior.

- **Risk band counts**  
  Quick snapshot of workload pressure (how many incidents need attention).

- **Top risky assignment groups**  
  Highlights teams where breaches happen more often.

- **Top risky categories**  
  Shows problem areas in the IT landscape.

- **Priority × Risk band heatmap**  
  Reveals mismatches (e.g., “Moderate priority but High risk”).

- **Calibration curve**  
  Confirms whether predicted probabilities align with real outcomes.

All visuals are explained inline in simple language inside the app.

---

## ▶️ Demo video (coming soon)

A short walkthrough video will be added here:

**Demo link:** _to be added_

The video will cover:
- Uploading the dataset
- Exploring the dashboard
- Interpreting risk and actions
- Downloading scored results

---

## 🚀 How to run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train the model
```bash
python run_train.py
```

### Run the app
```bash
streamlit run app.py
```

---

## 📤 Output

- Interactive dashboard (browser)
- Sortable and filterable incident list
- Incident‑level detail view
- Downloadable CSV with risk scores

---

## 📌 Project level

**Intermediate → Advanced**

This project demonstrates:
- Applied machine learning (not just theory)
- Feature engineering from event logs
- Probability calibration
- Decision‑oriented thinking
- End‑to‑end system design

---

## 🙌 Final note

This project was built to be **understandable**, **defensible**, and **useful**.  
Every decision is explainable, and every output is meant to support real operational choices.

