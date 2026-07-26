import os
import requests
import streamlit as st

# Where the FastAPI backend lives. Defaults to local; overridable when deployed.
API_URL = os.getenv("FINRISK_API_URL", "http://localhost:8000")

st.set_page_config(page_title="FinRisk Copilot", page_icon="🏦", layout="centered")

st.title("🏦 FinRisk Copilot")
st.caption("Credit-risk scoring, plain-English explanations, and banking-policy Q&A.")

# --- Sidebar: backend connection ---
with st.sidebar:
    st.subheader("Backend")
    st.write(f"API: `{API_URL}`")
    if st.button("Check connection"):
        try:
            r = requests.get(f"{API_URL}/health", timeout=5)
            r.raise_for_status()
            data = r.json()
            if data.get("model_loaded"):
                st.success("Connected · model loaded")
            else:
                st.warning("Connected · model NOT loaded")
        except Exception as e:
            st.error(f"Cannot reach API: {e}")

# --- Four tabs (filled in over the next steps) ---
tab_predict, tab_explain, tab_combined, tab_policy = st.tabs(
    ["Predict", "Explain", "Combined", "Ask Policy"]
)

# --- German Credit fields: human label -> dataset code ---
STATUS = {"No checking account": "A14", "< 0 DM": "A11",
          "0-200 DM": "A12", ">= 200 DM / salary assigned": "A13"}
CREDIT_HISTORY = {"No credits / all paid duly": "A30", "All paid at this bank": "A31",
                  "Existing credits paid duly": "A32", "Past delay in paying": "A33",
                  "Critical account / other credits": "A34"}
PURPOSE = {"Car (new)": "A40", "Car (used)": "A41", "Furniture/equipment": "A42",
           "Radio/TV": "A43", "Domestic appliances": "A44", "Repairs": "A45",
           "Education": "A46", "Retraining": "A48", "Business": "A49", "Other": "A410"}
SAVINGS = {"< 100 DM": "A61", "100-500 DM": "A62", "500-1000 DM": "A63",
           ">= 1000 DM": "A64", "Unknown / none": "A65"}
EMPLOYMENT = {"Unemployed": "A71", "< 1 year": "A72", "1-4 years": "A73",
              "4-7 years": "A74", ">= 7 years": "A75"}
PERSONAL = {"Male: single": "A93", "Male: divorced/separated": "A91",
            "Male: married/widowed": "A94", "Female: divorced/separated/married": "A92",
            "Female: single": "A95"}
DEBTORS = {"None": "A101", "Co-applicant": "A102", "Guarantor": "A103"}
PROPERTY = {"Real estate": "A121", "Life insurance / savings agreement": "A122",
            "Car or other": "A123", "Unknown / none": "A124"}
PLANS = {"None": "A143", "Bank": "A141", "Stores": "A142"}
HOUSING = {"Own": "A152", "Rent": "A151", "For free": "A153"}
JOB = {"Skilled employee / official": "A173", "Unskilled - resident": "A172",
       "Unemployed / unskilled - non-resident": "A171",
       "Management / self-employed / highly qualified": "A174"}
TELEPHONE = {"Yes, registered": "A192", "None": "A191"}
FOREIGN = {"Yes": "A201", "No": "A202"}


def application_form(prefix: str) -> dict:
    """Render the 20-field credit application form; return the API payload."""
    c1, c2 = st.columns(2)
    with c1:
        status = st.selectbox("Checking account status", STATUS, key=f"{prefix}_status")
        credit_history = st.selectbox("Credit history", CREDIT_HISTORY, key=f"{prefix}_ch")
        purpose = st.selectbox("Purpose", PURPOSE, key=f"{prefix}_purpose")
        savings = st.selectbox("Savings", SAVINGS, key=f"{prefix}_savings")
        employment = st.selectbox("Employment duration", EMPLOYMENT, key=f"{prefix}_emp")
        personal = st.selectbox("Personal status / sex", PERSONAL, key=f"{prefix}_personal")
        debtors = st.selectbox("Other debtors", DEBTORS, key=f"{prefix}_debtors")
        prop = st.selectbox("Property", PROPERTY, key=f"{prefix}_prop")
        plans = st.selectbox("Other installment plans", PLANS, key=f"{prefix}_plans")
        housing = st.selectbox("Housing", HOUSING, key=f"{prefix}_housing")
    with c2:
        job = st.selectbox("Job", JOB, key=f"{prefix}_job")
        telephone = st.selectbox("Telephone", TELEPHONE, key=f"{prefix}_tel")
        foreign = st.selectbox("Foreign worker", FOREIGN, key=f"{prefix}_foreign")
        duration = st.number_input("Duration (months)", 1, 120, 24, key=f"{prefix}_dur")
        amount = st.number_input("Credit amount (DM)", 1, 100000, 3500, key=f"{prefix}_amt")
        age = st.number_input("Age", 18, 100, 30, key=f"{prefix}_age")
        installment_rate = st.slider("Installment rate (% of income)", 1, 4, 2, key=f"{prefix}_rate")
        residence = st.number_input("Present residence since (years)", 1, 10, 2, key=f"{prefix}_res")
        n_credits = st.number_input("Existing credits at this bank", 1, 10, 1, key=f"{prefix}_ncred")
        liable = st.number_input("People liable to maintain", 1, 10, 1, key=f"{prefix}_liable")

    return {
        "status": STATUS[status], "duration": duration,
        "credit_history": CREDIT_HISTORY[credit_history], "purpose": PURPOSE[purpose],
        "amount": amount, "savings": SAVINGS[savings],
        "employment_duration": EMPLOYMENT[employment], "installment_rate": installment_rate,
        "personal_status_sex": PERSONAL[personal], "other_debtors": DEBTORS[debtors],
        "present_residence": residence, "property": PROPERTY[prop], "age": age,
        "other_installment_plans": PLANS[plans], "housing": HOUSING[housing],
        "number_credits": n_credits, "job": JOB[job], "people_liable": liable,
        "telephone": TELEPHONE[telephone], "foreign_worker": FOREIGN[foreign],
    }


# Class 1 means "bad credit" per the /explain docstring — see note below.
CLASS_LABELS = {0: "Good credit", 1: "Bad credit"}

with tab_predict:
    st.subheader("Credit risk score")
    features = application_form("pred")
    if st.button("Score application", key="pred_btn", type="primary"):
        try:
            r = requests.post(f"{API_URL}/predict", json=features, timeout=30)
            r.raise_for_status()
            data = r.json()
            pred, proba = data["prediction"], data["probabilities"]
            st.metric("Decision", CLASS_LABELS.get(pred, pred))
            st.write(f"Confidence: **{max(proba):.1%}**")
            st.progress(max(proba))
            with st.expander("Raw response"):
                st.json(data)
        except Exception as e:
            st.error(f"Request failed: {e}")
with tab_explain:
    st.info("Explain tab — coming soon.")
with tab_combined:
    st.info("Combined tab — coming soon.")
with tab_policy:
    st.info("Ask Policy tab — coming soon.")