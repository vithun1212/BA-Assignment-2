import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Churn & Revenue-at-Risk", page_icon="📉", layout="centered")

@st.cache_resource
def load():
    return (joblib.load("linear_model.sav"), joblib.load("logistic_model.sav"),
            joblib.load("model_meta.sav"))

lin, logit, meta = load()
opt = meta["cat_options"]

st.title("📉 Customer Churn & Revenue-at-Risk")
st.caption(f"Linear Regression (monthly charge): R² {meta['lin_metrics']['r2']}, "
           f"MAE {meta['lin_metrics']['mae']}  |  Logistic Regression (churn): "
           f"ROC-AUC {meta['log_metrics']['roc_auc']}, accuracy {meta['log_metrics']['accuracy']}")

st.subheader("1. Customer profile")
c1, c2 = st.columns(2)
with c1:
    gender = st.selectbox("Gender", opt["gender"])
    senior = st.selectbox("Senior citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", opt["Partner"])
    dependents = st.selectbox("Dependents", opt["Dependents"])
with c2:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract", opt["Contract"])
    paperless = st.selectbox("Paperless billing", opt["PaperlessBilling"])
    payment = st.selectbox("Payment method", opt["PaymentMethod"])

st.subheader("2. Service bundle")
c3, c4 = st.columns(2)
with c3:
    phone = st.selectbox("Phone service", opt["PhoneService"], index=1)
    multi = st.selectbox("Multiple lines", opt["MultipleLines"], disabled=(phone == "No"))
    internet = st.selectbox("Internet service", opt["InternetService"])
    no_net = internet == "No"
    security = st.selectbox("Online security", opt["OnlineSecurity"], disabled=no_net)
    backup = st.selectbox("Online backup", opt["OnlineBackup"], disabled=no_net)
with c4:
    device = st.selectbox("Device protection", opt["DeviceProtection"], disabled=no_net)
    tech = st.selectbox("Tech support", opt["TechSupport"], disabled=no_net)
    tv = st.selectbox("Streaming TV", opt["StreamingTV"], disabled=no_net)
    movies = st.selectbox("Streaming movies", opt["StreamingMovies"], disabled=no_net)

# add-ons are impossible without the base service
if phone == "No":
    multi = "No"
if no_net:
    security = backup = device = tech = tv = movies = "No"

services = {"PhoneService": phone, "MultipleLines": multi, "InternetService": internet,
            "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": device,
            "TechSupport": tech, "StreamingTV": tv, "StreamingMovies": movies}

# Step A: Linear Regression prices the bundle
est_monthly = max(0.0, float(lin.predict(pd.DataFrame([services]))[0]))
st.subheader("3. Charges")
st.info(f"Linear Regression estimate of monthly charge for this bundle: **{est_monthly:.2f}**")
monthly = st.number_input("Monthly charges (edit if actual bill is known)", 0.0, 200.0,
                          round(est_monthly, 2), step=0.5)

row = pd.DataFrame([{
    "SeniorCitizen": 1 if senior == "Yes" else 0, "tenure": tenure, "MonthlyCharges": monthly,
    "gender": gender, "Partner": partner, "Dependents": dependents, "Contract": contract,
    "PaperlessBilling": paperless, "PaymentMethod": payment, "InternetService": internet,
    "OnlineSecurity": security, "TechSupport": tech}])

# Step B: Logistic Regression scores churn; combine into revenue at risk
if st.button("Predict churn", type="primary"):
    p = float(logit.predict_proba(row)[0, 1])
    k1, k2, k3 = st.columns(3)
    k1.metric("Churn probability", f"{p:.1%}")
    k2.metric("Monthly revenue at risk", f"{p * monthly:.2f}")
    k3.metric("Annual revenue at risk", f"{p * monthly * 12:.2f}")
    st.progress(p)
    if p >= 0.5:
        st.error("High risk: prioritise for a retention offer.")
    else:
        st.success("Low risk: no immediate action needed.")

with st.expander("What drives churn? (Logistic Regression odds ratios)"):
    st.write("Odds ratio > 1 raises churn risk; < 1 lowers it.")
    st.write("**Top risk drivers**")
    st.dataframe(pd.Series(meta["top_risk"], name="Odds ratio"))
    st.write("**Top protective factors**")
    st.dataframe(pd.Series(meta["top_protect"], name="Odds ratio"))
