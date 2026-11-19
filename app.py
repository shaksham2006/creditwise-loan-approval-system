"""
app.py - CreditWise Loan Approval System (Streamlit dashboard).

Run:  python -m streamlit run app.py   (model.pkl must be in the same folder)
Works with the existing model.pkl made by train.py - no retraining needed.
"""
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CreditWise | Loan Approval", page_icon="🏦", layout="wide")

# ---------------------------------------------------------------- styling
st.markdown(
    """
<style>
.block-container {padding-top: 2.8rem; max-width: 1200px;}
.cw-hero {padding: 1.2rem 1.4rem; border-radius: 16px; margin-bottom: 1rem;
  background: linear-gradient(120deg, rgba(37,99,235,.18), rgba(16,185,129,.14));
  border: 1px solid rgba(128,128,128,.25);}
.cw-hero h1 {margin: 0 0 .2rem 0; font-size: 2rem; line-height: 1.2;}
.cw-hero p {margin: 0; opacity: .8;}
.cw-pill {display:inline-block; padding: .15rem .65rem; margin: .6rem .4rem 0 0; border-radius: 999px;
  font-size: .78rem; border: 1px solid rgba(128,128,128,.35); background: rgba(128,128,128,.10);}
.cw-card {border: 1px solid rgba(128,128,128,.25); border-radius: 16px; padding: 1rem 1.2rem;
  background: rgba(128,128,128,.06);}
.cw-badge {display:inline-block; padding: .35rem 1.1rem; border-radius: 999px; font-weight: 700;
  font-size: 1.05rem; color: #fff;}
.cw-ok {background:#059669;} .cw-no {background:#dc2626;}
.cw-bar-row {display:flex; align-items:center; margin: .28rem 0; font-size: .88rem;}
.cw-bar-label {width: 38%; opacity:.85;}
.cw-bar-track {flex:1; position:relative; height: 16px; background: rgba(128,128,128,.12); border-radius: 8px;}
.cw-bar-mid {position:absolute; left:50%; top:-2px; bottom:-2px; width:1px; background: rgba(128,128,128,.6);}
.cw-bar-fill {position:absolute; top:0; height:100%; border-radius: 8px;}
.cw-bar-val {width: 52px; text-align:right; font-variant-numeric: tabular-nums; opacity:.8;}
.cw-small {font-size:.8rem; opacity:.7;}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- model + logic
def add_features(d: pd.DataFrame) -> pd.DataFrame:
    """Same feature engineering as train.py."""
    d = d.copy()
    d["DTI_Ratio_sq"] = d["DTI_Ratio"] ** 2
    d["Credit_Score_sq"] = d["Credit_Score"] ** 2
    return d.drop(columns=["DTI_Ratio", "Credit_Score"])


@st.cache_resource
def load_bundle():
    return joblib.load("model.pkl")


def pretty(name: str) -> str:
    return name.replace("_", " ")


def feature_effects(model, row: pd.DataFrame, cat_cols):
    """Per-feature push on the approval log-odds (Logistic Regression only).

    Numeric features: effect versus an average applicant (scaled value x coefficient).
    Categories: effect versus the baseline category of that field.
    Returns a list of (label, effect) or None if the model has no coefficients.
    """
    clf = model[-1]
    if not hasattr(clf, "coef_"):
        return None
    prep = model[:-1]
    names = prep.get_feature_names_out()
    x = prep.transform(row)[0]
    coefs = clf.coef_[0]
    effects = {}
    for n, xv, c in zip(names, x, coefs):
        base = n.split("__", 1)[1]
        if n.startswith("num__"):
            label = {"DTI_Ratio_sq": "DTI_Ratio", "Credit_Score_sq": "Credit_Score"}.get(base, base)
        else:
            label = next((c0 for c0 in cat_cols if base.startswith(c0 + "_")), base)
        effects[label] = effects.get(label, 0.0) + float(xv * c)
    return [(pretty(k), v) for k, v in effects.items()]


def gauge_svg(prob: float) -> str:
    color = "#059669" if prob >= 0.5 else "#dc2626"
    arc = 251.33  # length of the half circle with radius 80
    return (
        '<svg viewBox="0 0 200 118" style="width:100%;max-width:340px;display:block;margin:auto">'
        '<path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="rgba(128,128,128,.25)" '
        'stroke-width="16" stroke-linecap="round"/>'
        f'<path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="{color}" stroke-width="16" '
        f'stroke-linecap="round" stroke-dasharray="{prob * arc:.1f} {arc}"/>'
        '<line x1="100" y1="12" x2="100" y2="30" stroke="rgba(128,128,128,.9)" stroke-width="2"/>'
        f'<text x="100" y="92" text-anchor="middle" font-size="34" font-weight="700" fill="{color}">{prob:.0%}</text>'
        '<text x="100" y="112" text-anchor="middle" font-size="9.5" fill="gray">approval probability</text>'
        "</svg>"
    )


def bars_html(items, scale=None, signed=True, fmt="{:+.1f}") -> str:
    """Horizontal bars. signed=True: centered, green right / red left. Otherwise left to right."""
    if not items:
        return ""
    scale = scale or max(abs(v) for _, v in items) or 1.0
    rows = []
    for label, v in items:
        w = min(abs(v) / scale, 1.0)
        if signed:
            color = "#059669" if v >= 0 else "#dc2626"
            style = f"left:50%;width:{w * 50:.1f}%" if v >= 0 else f"right:50%;width:{w * 50:.1f}%"
            mid = '<div class="cw-bar-mid"></div>'
        else:
            color, style, mid = "#2563eb", f"left:0;width:{w * 100:.1f}%", ""
        rows.append(
            f'<div class="cw-bar-row"><div class="cw-bar-label">{label}</div>'
            f'<div class="cw-bar-track">{mid}<div class="cw-bar-fill" style="{style};background:{color}"></div></div>'
            f'<div class="cw-bar-val">{fmt.format(v)}</div></div>'
        )
    return "".join(rows)


bundle = load_bundle()
model = bundle["model"]
info = bundle["form_info"]
results = pd.DataFrame(bundle["results"]).T
best = bundle["best_name"]
num_info, cat_info = info["numeric"], info["categorical"]

# ---------------------------------------------------------------- form state + presets
SLIDERS = {"Credit_Score", "DTI_Ratio", "Age", "Dependents", "Existing_Loans", "Loan_Term"}
GROUPS = {
    "👤 Applicant": ["Age", "Dependents", "Marital_Status", "Education_Level",
                     "Employment_Status", "Employer_Category", "Property_Area"],
    "💰 Financial profile": ["Applicant_Income", "Coapplicant_Income", "Credit_Score", "DTI_Ratio",
                             "Existing_Loans", "Savings", "Collateral_Value"],
    "🏠 Loan details": ["Loan_Amount", "Loan_Term", "Loan_Purpose"],
}
known = {c for cols in GROUPS.values() for c in cols}
leftovers = [c for c in list(num_info) + list(cat_info) if c not in known]
if leftovers:
    GROUPS["Other"] = leftovers


def typical():
    v = {c: (int(m["median"]) if m["is_int"] else float(m["median"])) for c, m in num_info.items()}
    v.update({c: opts[0] for c, opts in cat_info.items()})
    return v


def clamp(col, val):
    m = num_info[col]
    val = min(max(val, m["min"]), m["max"])
    return int(val) if m["is_int"] else float(val)


def pick(col, wanted):
    opts = cat_info[col]
    return wanted if wanted in opts else opts[0]


def preset(kind):
    v = typical()
    if kind == "strong":
        num = dict(Credit_Score=780, DTI_Ratio=0.2, Applicant_Income=14000, Savings=15000,
                   Loan_Amount=8000, Existing_Loans=0, Age=38)
        cat = dict(Employment_Status="Salaried", Employer_Category="Government", Education_Level="Graduate",
                   Loan_Purpose="Car", Marital_Status="Married")
    elif kind == "weak":
        num = dict(Credit_Score=560, DTI_Ratio=0.55, Applicant_Income=2500, Savings=500,
                   Loan_Amount=35000, Existing_Loans=4, Age=24)
        cat = dict(Employment_Status="Unemployed", Employer_Category="Unemployed", Education_Level="Not Graduate",
                   Loan_Purpose="Personal", Marital_Status="Single")
    else:
        return v
    v.update({c: clamp(c, x) for c, x in num.items() if c in num_info})
    v.update({c: pick(c, x) for c, x in cat.items() if c in cat_info})
    return v


def apply_preset(kind):
    for c, val in preset(kind).items():
        st.session_state[f"in_{c}"] = val


for c, val in typical().items():
    st.session_state.setdefault(f"in_{c}", val)


def widget(col, target):
    key, label = f"in_{col}", pretty(col)
    if col in num_info:
        m = num_info[col]
        if m["is_int"]:
            lo, hi, step = int(m["min"]), int(m["max"]), 1
        else:
            lo, hi, step = float(m["min"]), float(m["max"]), 0.01
        if col in SLIDERS:
            return target.slider(label, lo, hi, key=key, step=step)
        return target.number_input(label, lo, hi, key=key, step=step)
    return target.selectbox(label, cat_info[col], key=key)


# ---------------------------------------------------------------- header
st.markdown(
    f"""
<div class="cw-hero">
<h1>🏦 CreditWise Loan Approval System</h1>
<p>Machine-learning model that estimates the probability a loan application is approved.</p>
<span class="cw-pill">Model: {best}</span>
<span class="cw-pill">Test accuracy: {results.loc[best, 'Accuracy']:.0%}</span>
<span class="cw-pill">F1: {results.loc[best, 'F1']:.2f}</span>
<span class="cw-pill">Gender excluded to reduce bias</span>
</div>
""",
    unsafe_allow_html=True,
)

tab_predict, tab_models, tab_about = st.tabs(["🔮 Predict", "📊 Model insights", "ℹ️ About"])

# ---------------------------------------------------------------- tab 1: predict
with tab_predict:
    b1, b2, b3, _ = st.columns([1.1, 1.1, 1, 3])
    b1.button("✅ Strong applicant", on_click=apply_preset, args=("strong",), use_container_width=True)
    b2.button("❌ Weak applicant", on_click=apply_preset, args=("weak",), use_container_width=True)
    b3.button("↺ Reset", on_click=apply_preset, args=("reset",), use_container_width=True)

    form_col, result_col = st.columns([1.35, 1], gap="large")

    values = {}
    with form_col:
        for title, cols in GROUPS.items():
            with st.container(border=True):
                st.markdown(f"**{title}**")
                grid = st.columns(2)
                for i, c in enumerate(cols):
                    values[c] = widget(c, grid[i % 2])

    row = add_features(pd.DataFrame([values]))
    prob = float(model.predict_proba(row)[0][1])
    approved = prob >= 0.5
    margin = abs(prob - 0.5)
    confidence = "High confidence" if margin > 0.35 else "Moderate confidence" if margin > 0.15 else "Borderline case"

    with result_col:
        st.markdown(
            '<div class="cw-card" style="text-align:center">'
            + gauge_svg(prob)
            + f'<div style="margin-top:.6rem"><span class="cw-badge {"cw-ok" if approved else "cw-no"}">'
            + ("Likely APPROVED" if approved else "Likely REJECTED")
            + f'</span></div><div class="cw-small" style="margin-top:.5rem">{confidence} &middot; decision threshold 50%</div></div>',
            unsafe_allow_html=True,
        )

        effects = feature_effects(model, row, list(cat_info))
        st.markdown("&nbsp;")
        st.markdown("**Why this result?**")
        if effects:
            top = sorted(effects, key=lambda t: abs(t[1]), reverse=True)[:8]
            st.markdown(
                '<div class="cw-card">' + bars_html(top)
                + '<div class="cw-small" style="margin-top:.5rem">Green pushes toward approval, red toward rejection '
                "(log-odds, vs. an average applicant; categories vs. their baseline).</div></div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("Feature effects are available for Logistic Regression models only.")

    st.caption("Student ML project for learning purposes. Not financial advice.")

# ---------------------------------------------------------------- tab 2: model insights
with tab_models:
    st.subheader("Three models, compared")
    st.write("Each model was trained on 80% of the data and scored on the held-out 20%. "
             "Preprocessing is fitted on the training split only, so nothing leaks from the test set.")

    cols = st.columns(len(results))
    for col, (name, r) in zip(cols, results.iterrows()):
        col.metric(name + (" ⭐" if name == best else ""), f"F1 {r['F1']:.2f}", f"Accuracy {r['Accuracy']:.0%}", delta_color="off")

    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.markdown("**Full scores**")
        st.dataframe(results.round(3), use_container_width=True)
        st.markdown("**Test-set comparison**")
        st.bar_chart(results[["Precision", "Recall", "F1", "Accuracy"]])
    with right:
        clf = model[-1]
        if hasattr(clf, "coef_"):
            names = model[:-1].get_feature_names_out()
            imp = {}
            for n, c in zip(names, clf.coef_[0]):
                base = n.split("__", 1)[1]
                if n.startswith("num__"):
                    label = {"DTI_Ratio_sq": "DTI_Ratio", "Credit_Score_sq": "Credit_Score"}.get(base, base)
                else:
                    label = next((c0 for c0 in cat_info if base.startswith(c0 + "_")), base)
                imp[label] = imp.get(label, 0.0) + abs(float(c))
            ranked = sorted(((pretty(k), v) for k, v in imp.items()), key=lambda t: t[1], reverse=True)[:8]
            st.markdown(f"**What drives the {best} model**")
            st.markdown('<div class="cw-card">' + bars_html(ranked, signed=False, fmt="{:.2f}")
                        + '<div class="cw-small" style="margin-top:.5rem">Sum of absolute coefficients '
                        "per feature (inputs are standardised, so sizes are comparable).</div></div>",
                        unsafe_allow_html=True)
    st.caption(f"{best} was chosen by highest test F1. The 5-fold cross-validated F1 on the training data is "
               f"shown in the table as a stability check.")

# ---------------------------------------------------------------- tab 3: about
with tab_about:
    st.subheader("About this project")
    st.markdown(
        """
**Goal.** Predict loan approval from applicant, financial and loan details.

**Pipeline.**
1. Drop the ID and Gender columns (Gender is removed on purpose to avoid biased decisions).
2. Add squared Credit Score and DTI ratio features.
3. Split train/test first, then impute, scale and one-hot encode inside a scikit-learn `Pipeline`.
4. Compare Logistic Regression, KNN and Naive Bayes with precision, recall, F1, accuracy and 5-fold CV.
5. Save the best model and serve it with Streamlit.

**Limitations.** Small dataset, learning project only. It must not be used for real lending decisions.
"""
    )
