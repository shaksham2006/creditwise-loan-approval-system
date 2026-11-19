# CreditWise: Loan Approval Prediction System

A supervised machine learning project that predicts whether a loan application is likely to be **approved or rejected**, based on the applicant's financial and personal details. The best model is deployed as a web app built with Streamlit.

**Live demo:** https://creditwise-loan-approval-system-vk18.streamlit.app/

> This is a student learning project. It is not real financial advice and should not be used for actual lending decisions.

## Problem statement

Banks have to decide quickly whether to approve a loan. Given details such as income, credit score, debt-to-income ratio, savings and loan amount, can a model predict the outcome of the application? This project compares three classification models and deploys the best one.

## Dataset

- File: `loan_approval_data.csv`
- Source: *(add where you got it, e.g. Apna College course material / Kaggle link)*
- Target: `Loan_Approved` (approved / rejected)
- Inputs: 17 features used by the model

| Type | Features |
|---|---|
| Numeric | Applicant_Income, Coapplicant_Income, Age, Dependents, Credit_Score, Existing_Loans, DTI_Ratio, Savings, Collateral_Value, Loan_Amount, Loan_Term |
| Categorical | Employment_Status, Marital_Status, Loan_Purpose, Property_Area, Education_Level, Employer_Category |

`Gender` is deliberately excluded from the model to avoid gender-based bias in a lending decision. It can be switched back on with `DROP_GENDER = False` in `train.py`.

## Approach

1. Exploratory data analysis in `credit_wise.ipynb` (distributions, class balance, correlations).
2. Rows with a missing target (`Loan_Approved`) are dropped rather than filled in, so no labels are invented.
3. The data is split into train and test sets **before** any cleaning, encoding or scaling, to avoid data leakage.
4. Missing values are imputed, categorical features are one-hot encoded and numeric features are scaled, with each step fitted on the training data only.
5. Three classification models are trained and compared on the test set and with 5-fold cross-validation.
6. The best model is saved to `model.pkl` and loaded by the Streamlit app.

## Results

| Model | Accuracy | F1 score |
|---|---|---|
| Logistic Regression | 0.90 | 0.84 |
| *(Model 2: fill in from train.py output)* | | |
| *(Model 3: fill in from train.py output)* | | |

**Logistic Regression** performed best on F1 and accuracy, so it is the model used in the app.

Key finding: `Credit_Score` has by far the strongest relationship with approval (correlation about 0.45), far above income (about 0.12).

## Limitations

- The dataset is small, and the classes are imbalanced (more rejections than approvals).
- Results come from one dataset and have not been validated on real lending data.
- In a real setting, a false approval and a false rejection have different costs, so the metric and threshold would need to be chosen with that in mind.

## Run it locally

```bash
pip install -r requirements.txt
python train.py                  # trains the models and saves model.pkl
python -m streamlit run app.py   # opens the app at http://localhost:8501
```

## Project structure

```
├── app.py                  # Streamlit web app
├── train.py                # trains the models and saves model.pkl
├── model.pkl               # saved trained model
├── credit_wise.ipynb       # exploratory analysis and experiments
├── loan_approval_data.csv  # dataset
├── requirements.txt        # Python dependencies
└── README.md
```

## Tech stack

Python, pandas, NumPy, scikit-learn, Streamlit

## Author

Shaksham Verma · [GitHub](https://github.com/shaksham2006)
