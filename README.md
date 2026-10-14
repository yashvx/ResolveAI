# ResolveAI — AI-Powered Support Ticket Intelligence

ResolveAI is a full-stack machine learning application that classifies customer support tickets, recommends priorities, flags uncertain predictions for human review, and provides a dashboard to manage tickets and monitor support activity.

## Problem Statement

Customer support teams receive messages about refunds, payments, technical problems, account access, and shipping. Manually categorizing every message can be time-consuming and inconsistent.

ResolveAI explores how machine learning can assist this workflow by predicting ticket categories and surfacing uncertain predictions for human review.

## Features

- **Automatic ticket classification:** Predicts one of five categories: Refund, Payment, Technical, Account, or Shipping.
- **Confidence-based review flags:** Marks predictions below a provisional confidence threshold for human review.
- **Priority recommendations:** Uses transparent keyword rules to flag certain urgent or security-related messages as high priority.
- **Ticket management:** Create tickets, search and filter tickets, and update their statuses.
- **Analytics dashboard:** Summarizes ticket counts, statuses, high-priority tickets, review flags, and category distribution.
- **REST API:** FastAPI endpoints with interactive API documentation.
- **Automated tests:** Backend tests cover API behavior and priority recommendation rules.

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | React, Vite, JavaScript |
| UI and charts | CSS, Recharts, Lucide React |
| Backend | Python, FastAPI |
| Machine learning | scikit-learn, TF-IDF, Logistic Regression |
| Database | SQLite, SQLAlchemy |
| Model persistence | Joblib |
| Testing | Pytest, FastAPI TestClient |

## Architecture

```text
React Dashboard
      |
      | HTTP requests
      v
FastAPI Backend
      |
      +---- Ticket classification pipeline
      |       TF-IDF + Logistic Regression
      |
      +---- Keyword-based priority rules
      |
      +---- SQLAlchemy
                 |
                 v
             SQLite Database
```

The trained classification pipeline is stored at `backend/ml/ticket_classifier.joblib` and loaded by the backend when the application starts.

## Ticket Classification

The current model uses a scikit-learn pipeline with:

1. **TF-IDF vectorization** to represent ticket text as numerical features.
2. **Unigram and bigram features** to capture individual words and two-word phrases.
3. **Logistic Regression** to predict the ticket category.
4. **Prediction confidence** to identify cases that may need human review.

The current training dataset contains 120 handcrafted synthetic examples across five categories. These examples are useful for demonstrating the workflow, but they do not represent a production dataset.

The review threshold is currently set to `0.55`. It is provisional and should be evaluated and calibrated on representative validation data before real-world use. Model confidence is not automatically a calibrated probability of correctness.

Priority recommendations are separate from the ML category prediction and use explicit keyword rules.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API welcome message |
| GET | `/health` | Health check |
| POST | `/predict` | Classify a ticket message |
| POST | `/tickets` | Create and classify a ticket |
| GET | `/tickets` | List tickets, optionally filtered by status or category |
| PATCH | `/tickets/{ticket_id}` | Update ticket status |
| GET | `/analytics/summary` | Retrieve dashboard analytics |

Interactive API documentation is available at `/docs` when the backend is running.

## Getting Started

### Prerequisites

- Python 3.11 or later
- Node.js and npm
- Git

### 1. Clone the repository

```bash
git clone https://github.com/yashvx/ResolveAI.git
cd ResolveAI
```

### 2. Start the backend

In a terminal:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
PYTHONPATH=. python -m uvicorn app.main:app --reload
```

The API will be available at:

- API: http://127.0.0.1:8000
- Interactive docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

Keep the backend terminal running.

### 3. Start the frontend

Open a second terminal:

```bash
cd ~/Desktop/ResolveAI/frontend
npm install
npm run dev
```

Open the local URL printed by Vite, usually http://localhost:5173.

### 4. Run backend tests

In a terminal:

```bash
cd ~/Desktop/ResolveAI/backend
source .venv/bin/activate
PYTHONPATH=. python -m pytest -q tests/
```

### 5. Evaluate the model

From the backend directory, with the virtual environment activated:

Run `PYTHONPATH=.:ml python ml/evaluate_confidence.py` to evaluate confidence and classification performance.

Run `PYTHONPATH=.:ml python ml/evaluate_holdout.py` to evaluate the first synthetic holdout dataset.

Run `PYTHONPATH=.:ml python ml/evaluate_calibration.py` to inspect confidence calibration and review-threshold trade-offs.

A second manually authored dataset is available at `ml/holdout_examples_day6.json`. It produced 25 correct predictions out of 25 in the recorded evaluation. This small synthetic result does not establish expected real-world accuracy. Calibration and threshold results are exploratory.

### 6. Retrain the model

From the backend directory, with the virtual environment activated:

```bash
PYTHONPATH=.:ml python -m ml.train_model
```

Retraining overwrites the saved model file. Restart the backend afterward so it loads the updated model.

## Current Limitations

- The training dataset is small and synthetic.
- Evaluation results may not generalize to real customer support messages.
- The confidence threshold has not yet been validated on a representative production dataset.
- Priority recommendations are rule-based and may miss urgent cases or flag non-urgent ones.
- The application is a learning and portfolio project, not a production-ready support system.

## Roadmap

- Improve evaluation using stratified validation and more realistic unseen examples.
- Analyze per-category precision, recall, F1-score, and confusion matrices.
- Evaluate confidence thresholds and human-review workload.
- Improve input validation, error handling, and automated test coverage.
- Add deployment instructions and application screenshots.

## Project Status

The core full-stack workflow is implemented. The backend test suite and frontend production build have passed in the development environment.

## License

No license has been added yet. All rights are reserved unless a license is added to this repository.
