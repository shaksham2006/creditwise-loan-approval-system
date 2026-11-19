"""
train.py - trains the CreditWise loan approval models and saves the best one.

Run once:  python train.py
Output:    model.pkl  (used by app.py)
"""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_FILE = "loan_approval_data.csv"
DROP_GENDER = True  # set to False if you want Gender used as a feature again


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering from the notebook (same function is used in app.py)."""
    d = d.copy()
    d["DTI_Ratio_sq"] = d["DTI_Ratio"] ** 2
    d["Credit_Score_sq"] = d["Credit_Score"] ** 2
    return d.drop(columns=["DTI_Ratio", "Credit_Score"])


# ---------- 1. Load data ----------
df = pd.read_csv(DATA_FILE)
df = df.drop(columns=["Applicant_ID"])
if DROP_GENDER:
    df = df.drop(columns=["Gender"])

df = df.dropna(subset=["Loan_Approved"])
y = df["Loan_Approved"].map({"No": 0, "Yes": 1})
X_raw = df.drop(columns=["Loan_Approved"])

# Info the app needs to build its input form
raw_num_cols = X_raw.select_dtypes(include="number").columns.tolist()
raw_cat_cols = [c for c in X_raw.columns if c not in raw_num_cols]
form_info = {
    "numeric": {
        c: {
            "min": float(X_raw[c].min()),
            "max": float(X_raw[c].max()),
            "median": float(X_raw[c].median()),
            "is_int": bool((X_raw[c].dropna() % 1 == 0).all()),
        }
        for c in raw_num_cols
    },
    "categorical": {c: sorted(X_raw[c].dropna().unique().tolist()) for c in raw_cat_cols},
}

# ---------- 2. Feature engineering ----------
X = add_features(X_raw)
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

# ---------- 3. Split FIRST (so nothing from the test set leaks into training) ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- 4. Preprocessing (fitted on training data only, inside the pipeline) ----------
preprocess = ColumnTransformer(
    [
        ("num", Pipeline([("impute", SimpleImputer(strategy="mean")), ("scale", StandardScaler())]), num_cols),
        (
            "cat",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
                ]
            ),
            cat_cols,
        ),
    ]
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
}

# ---------- 5. Train + evaluate all 3 models ----------
results = {}
pipelines = {}
for name, clf in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", clf)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    cv_f1 = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1").mean()
    results[name] = {
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "Accuracy": accuracy_score(y_test, pred),
        "CV F1 (5-fold)": cv_f1,
    }
    pipelines[name] = pipe
    print(f"\n{name}")
    for k, v in results[name].items():
        print(f"  {k:<15}: {v:.3f}")
    print("  Confusion matrix:\n", confusion_matrix(y_test, pred))

# ---------- 6. Pick the best model by F1 and save it ----------
best_name = max(results, key=lambda n: results[n]["F1"])
print(f"\nBest model by F1: {best_name}")

joblib.dump(
    {
        "model": pipelines[best_name],
        "best_name": best_name,
        "results": results,
        "form_info": form_info,
    },
    "model.pkl",
)
print("Saved model.pkl")
