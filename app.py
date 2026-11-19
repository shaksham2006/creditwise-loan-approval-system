"""
app.py - Streamlit web app for the CreditWise Loan Approval System.

Run:  streamlit run app.py   (after running train.py once)
"""
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CreditWise Loan System", page_icon="🏦", layout="centered")


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    """Same feature engineering as train.py."""
    d = d.copy()
    d["DTI_Ratio_sq"] = d["DTI_Ratio"] ** 2
    d["Credit_Score_sq"] = d["Credit_Score"] ** 2
    return d.drop(columns=["DTI_Ratio", "Credit_Score"])


@st.cache_resource
def load_bundle():
    return joblib.load("model.pkl")


bundle = load_bundle()
model = bundle["model"]
info = bundle["form_info"]

st.title("🏦 CreditWise Loan Approval System")
st.write(
    "Enter the applicant's details and the model will predict whether the loan "
    "is likely to be **approved**. Built with scikit-learn and Streamlit."
)

with st.form("loan_form"):
    values = {}
    left, right = st.columns(2)

    # Numeric inputs (ranges come from the training data)
    for i, (col, meta) in enumerate(info["numeric"].items()):
        target = left if i % 2 == 0 else right
        label = col.replace("_", " ")
        if meta["is_int"]:
            values[col] = target.number_input(
                label,
                min_value=int(meta["min"]),
                max_value=int(meta["max"]),
                value=int(meta["median"]),
                step=1,
            )
        else:
            values[col] = target.number_input(
                label,
                min_value=float(meta["min"]),
                max_value=float(meta["max"]),
                value=float(meta["median"]),
            )

    # Dropdown inputs
    for i, (col, options) in enumerate(info["categorical"].items()):
        target = left if i % 2 == 0 else right
        values[col] = target.selectbox(col.replace("_", " "), options)

    submitted = st.form_submit_button("Check eligibility", use_container_width=True)

if submitted:
    row = add_features(pd.DataFrame([values]))
    prob = float(model.predict_proba(row)[0][1])

    st.subheader("Result")
    if prob >= 0.5:
        st.success(f"✅ Likely APPROVED  (approval probability: {prob:.0%})")
    else:
        st.error(f"❌ Likely REJECTED  (approval probability: {prob:.0%})")
    st.progress(prob)
    st.caption("This is a student ML project for learning purposes, not real financial advice.")

with st.expander(f"About the model ({bundle['best_name']})"):
    st.write("Test-set performance of the three models I trained:")
    st.dataframe(pd.DataFrame(bundle["results"]).T.round(3))
