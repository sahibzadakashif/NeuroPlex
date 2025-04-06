# app.py
import streamlit as st
import pandas as pd
import numpy as np
import base64
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem

# Load the model (after retraining with train_model.py)
model = joblib.load("model.pkl")

# Page configuration
st.set_page_config(page_title='NeuroPlex', layout='wide', page_icon='🧠')

# Utility to convert SMILES to Morgan fingerprint
def smiles_to_morgan_fp(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits))
    return None

# Predict pIC50 and classify
def predict_from_smiles(smiles_list):
    results = []
    for smi in smiles_list:
        fp = smiles_to_morgan_fp(smi)
        if fp is not None:
            pIC50 = model.predict([fp])[0]
            if pIC50 >= 6:
                activity = "Active"
            elif pIC50 >= 5:
                activity = "Intermediate"
            else:
                activity = "Inactive"
            results.append((smi, round(pIC50, 2), activity))
        else:
            results.append((smi, None, "Invalid SMILES"))
    return pd.DataFrame(results, columns=["SMILES", "Predicted pIC50", "Bioactivity Class"])

# CSV download helper
def download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">📥 Download CSV</a>'

# App UI
def main():
    st.markdown("<h1 style='text-align: center; color: #7A4E9F;'>🧬 NeuroPlex</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>AI-based pIC₅₀ prediction for Alzheimer's drug discovery</p>", unsafe_allow_html=True)

    input_method = st.radio("Input Method", ["Paste SMILES", "Upload File"])

    if input_method == "Paste SMILES":
        smiles_text = st.text_area("Enter SMILES (one per line)")
        if st.button("Predict"):
            smiles_list = [s.strip() for s in smiles_text.strip().splitlines() if s.strip()]
            if not smiles_list:
                st.warning("Please enter valid SMILES.")
            else:
                results = predict_from_smiles(smiles_list)
                st.success("✅ Predictions ready!")
                st.dataframe(results)
                st.markdown(download_link(results), unsafe_allow_html=True)

    else:
        uploaded = st.file_uploader("Upload CSV or TXT", type=["csv", "txt"])
        if uploaded and st.button("Predict"):
            try:
                df = pd.read_csv(uploaded, header=None)
                smiles_list = df.iloc[:, 0].dropna().astype(str).tolist()
                results = predict_from_smiles(smiles_list)
                st.success("✅ Predictions ready!")
                st.dataframe(results)
                st.markdown(download_link(results), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown("#### 👨‍🔬 Team NeuroPlex")
    st.markdown("""
    - **Dr. Kashif Iqbal Sahibzada** – UOL, Pakistan & HAUT, China  
    - **Dr. Andleeb Batool** – GCU, Lahore  
    - **Shumaila Shahid** – PU, Lahore
    """)

if __name__ == "__main__":
    main()


    st.markdown("---")
    show_team()

if __name__ == "__main__":
    main()
