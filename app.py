# neuroplex_app.py

import streamlit as st
import pandas as pd
import numpy as np
import base64
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem

# Load your trained model
model = joblib.load("model2.pkl")  # Make sure this exists in your app folder

# Page config
st.set_page_config(
    page_title='NeuroPlex',
    layout='wide',
    initial_sidebar_state='expanded',
    page_icon='🧠',
)

# ---------- Utility Functions ----------
def smiles_to_morgan_fp(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits))

def predict_from_smiles(smiles_list):
    predictions = []
    for smi in smiles_list:
        fp = smiles_to_morgan_fp(smi)
        if fp is None:
            predictions.append((smi, None, "Invalid SMILES"))
        else:
            pIC50 = model.predict([fp])[0]
            if pIC50 >= 6:
                activity = "Active"
            elif pIC50 >= 5:
                activity = "Intermediate"
            else:
                activity = "Inactive"
            predictions.append((smi, round(pIC50, 2), activity))
    return pd.DataFrame(predictions, columns=["SMILES", "Predicted pIC50", "Bioactivity Class"])

def download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">📥 Download Results</a>'

# ---------- UI Components ----------
def show_header():
    st.markdown("""
        <h1 style='text-align: center; color: #7A4E9F;'>🧬 NeuroPlex – AI-Driven Drug Discovery</h1>
        <p style='text-align: center; font-size: 16px;'>
        Predict pIC₅₀ values and bioactivity classes of potential drug candidates for Alzheimer's Disease.
        </p>
    """, unsafe_allow_html=True)

def show_team():
    st.markdown("### 👨‍🔬 Team NeuroPlex")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **Dr. Kashif Iqbal Sahibzada**  
        Assistant Professor, DHPT, UOL  
        Post-Doctoral Fellow, HAUT China  
        📧 kashif.iqbal@dhpt.uol.edu.pk
        """)

    with col2:
        st.markdown("""
        **Dr. Andleeb Batool**  
        Assistant Professor, Dept. of Zoology, GCU Lahore  
        📧 andleeb.batool@gcu.edu.pk
        """)

    with col3:
        st.markdown("""
        **Shumaila Shahid**  
        MS Biochemistry, SBB, PU Lahore  
        📧 shumaila.ms.sbb@pu.edu.pk
        """)

# ---------- Main App ----------
def main():
    show_header()

    st.markdown("### 🔍 Enter SMILES")
    input_method = st.radio("Input Method", ["Paste SMILES", "Upload File"])

    if input_method == "Paste SMILES":
        smiles_text = st.text_area("Enter one or more SMILES (one per line):")
        if st.button("Predict"):
            smiles_list = [s.strip() for s in smiles_text.strip().split("\n") if s.strip()]
            if not smiles_list:
                st.warning("Please enter at least one valid SMILES.")
                return
            result_df = predict_from_smiles(smiles_list)
            st.success("✅ Prediction completed.")
            st.dataframe(result_df)
            st.markdown(download_link(result_df), unsafe_allow_html=True)

    else:  # Upload file
        uploaded_file = st.file_uploader("Upload a CSV or TXT file with SMILES", type=["csv", "txt"])
        if uploaded_file and st.button("Predict"):
            try:
                df = pd.read_csv(uploaded_file, header=None)
                smiles_list = df.iloc[:, 0].dropna().astype(str).tolist()
                result_df = predict_from_smiles(smiles_list)
                st.success("✅ Prediction completed.")
                st.dataframe(result_df)
                st.markdown(download_link(result_df), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")

    st.markdown("---")
    show_team()

if __name__ == "__main__":
    main()
