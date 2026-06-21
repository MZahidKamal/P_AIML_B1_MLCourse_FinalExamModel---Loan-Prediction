# 🏦 Loan Approval Prediction System

A machine learning web application that predicts whether a bank loan application will be **approved** or **rejected** based on applicant information.

Built as part of the **ML Final Exam — Phitron Batch 01**.

---

## 🚀 Live Demo

**Hugging Face Deployment:** [Click here to try the app](https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME)

> Replace the link above with your actual Hugging Face Spaces URL after deployment.

---

## 📋 Project Overview

| Item | Details |
|---|---|
| **Problem Type** | Binary Classification |
| **Target** | Loan Approved (Y) / Rejected (N) |
| **Dataset** | Loan Prediction Dataset (Kaggle) |
| **Primary Model** | Logistic Regression |
| **Tuning Method** | GridSearchCV |
| **Web Interface** | Gradio |
| **Deployment** | Hugging Face Spaces |

---

## 📁 Repository Structure

```
├── ML_Final_Exam_Loan_Approval.ipynb   # Main notebook (all 11 tasks)
├── app.py                               # Gradio web app
├── best_loan_model.pkl                  # Trained and saved model
├── requirements.txt                     # Python dependencies
├── loan-predication-dataset.csv         # Dataset
└── README.md                            # This file
```

---

## 🔧 Tasks Completed

| # | Task | Marks |
|---|---|---|
| 1 | Data Loading | 5 |
| 2 | Data Preprocessing (6 steps) | 10 |
| 3 | Pipeline Creation | 10 |
| 4 | Primary Model Selection | 5 |
| 5 | Model Training | 10 |
| 6 | Cross-Validation (5-Fold Stratified) | 10 |
| 7 | Hyperparameter Tuning (GridSearchCV) | 10 |
| 8 | Best Model Selection | 10 |
| 9 | Model Performance Evaluation | 10 |
| 10 | Web Interface with Gradio | 10 |
| 11 | Deployment to Hugging Face | 10 |
| | **Total** | **100** |

---

## 🧠 Model Details

**Algorithm:** Logistic Regression

**Why Logistic Regression?**
- Binary classification problem (Approved/Rejected)
- Interpretable coefficients — important for financial decisions
- Works well with mixed numeric + categorical features after preprocessing
- Handles class imbalance via `class_weight='balanced'`

**Preprocessing Pipeline:**
- Numeric columns (`LoanAmount`, `Loan_Amount_Term`, `Credit_History`, `Total_Income`) → Median Imputation + StandardScaler
- Categorical columns (`Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `Property_Area`) → Most Frequent Imputation + OneHotEncoder

**Hyperparameter Tuning:**
- Tuned: `C`, `solver`, `penalty`
- Method: GridSearchCV with 5-Fold Stratified Cross-Validation
- Scoring metric: F1 Score (chosen due to class imbalance)

---

## 📊 Dataset

**Source:** [Loan Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/ninzaami/loan-predication)

| Property | Value |
|---|---|
| Rows | 614 |
| Columns | 13 |
| Target | `Loan_Status` (Y/N) |
| Class Distribution | ~69% Approved, ~31% Rejected |

**Features used:**

| Feature | Type | Description |
|---|---|---|
| Gender | Categorical | Male / Female |
| Married | Categorical | Yes / No |
| Dependents | Categorical | 0 / 1 / 2 / 3+ |
| Education | Categorical | Graduate / Not Graduate |
| Self_Employed | Categorical | Yes / No |
| LoanAmount | Numeric | Loan amount in thousands |
| Loan_Amount_Term | Numeric | Term in months |
| Credit_History | Numeric | 1 = Good, 0 = Bad |
| Property_Area | Categorical | Urban / Semiurban / Rural |
| Total_Income | Numeric (engineered) | ApplicantIncome + CoapplicantIncome |

---

## 🖥️ How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Gradio App

```bash
python app.py
```

The app will open at `http://127.0.0.1:7860` in your browser.

### 5. Run the Notebook

Open `ML_Final_Exam_Loan_Approval.ipynb` in Jupyter or Google Colab and run all cells in order.

> **Note:** Place `loan-predication-dataset.csv` in the same folder as the notebook before running.

---

## ☁️ How to Deploy on Hugging Face Spaces

1. Go to [huggingface.co](https://huggingface.co) and log in.
2. Click **New Space** → Select **Gradio** as the SDK.
3. Set visibility to **Public**.
4. Upload these files:
   - `app.py`
   - `requirements.txt`
   - `best_loan_model.pkl`
5. Hugging Face will automatically install dependencies and launch the app.
6. Copy the public URL from the Space page.

---

## 📦 Requirements

Key dependencies:

```
gradio
scikit-learn
pandas
numpy
```

Full list available in `requirements.txt`.

---

## 👤 Author

**Name:** Mohammad Zahid Kamal
**Email:** kamal.md.zahid@gmail.com
**Course:** Phitron — AI/ML Expert Track, Batch 01
