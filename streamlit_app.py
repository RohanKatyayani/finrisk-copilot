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

with tab_predict:
    st.info("Predict tab — coming next.")
with tab_explain:
    st.info("Explain tab — coming soon.")
with tab_combined:
    st.info("Combined tab — coming soon.")
with tab_policy:
    st.info("Ask Policy tab — coming soon.")