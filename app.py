# app.py
import streamlit as st
import pandas as pd
import numpy as np
import base64
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem

# Load model
model = joblib.load("model.pkl")

st.set_page_config(page_title="NeuroPlex", layout="wide")

# Inject custom CSS
st.markdown("""
    <style>
    body {
        background-color: #ffffff;
        color: #000000;
    }
    .stApp {
        background-color: #ffffff;
        color: #000000;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #006a4e; /* Bottle green */
    }
    .stRadio > div {
        color: #000000;
    }
    .stButton>button {
        background-color: #006a4e;
        color: white;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #228b73;
        color: #ffffff;
    }
    .stTextArea, .stFileUploader {
        background-color: #f0f0f0;
        color: black;
        border-radius: 10px;
    }
    .stDataFrame {
        background-color: #ffffff;
        color: black;
    }
    a {
        color: #006a4e;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Helper Functions
def smiles_to_morgan_fp(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits))
    return None

def predict_pIC50_and_class(smiles_list):
    results = []
    for smi in smiles_list:
        fp = smiles_to_morgan_fp(smi)
        if fp is not None:
            pIC50 = model.predict([fp])[0]
            activity = (
                "🟢 Active" if pIC50 >= 6 else
                "🟡 Intermediate" if pIC50 >= 5 else
                "🔴 Inactive"
            )
            results.append((smi, round(pIC50, 2), activity))
        else:
            results.append((smi, None, "❌ Invalid SMILES"))
    return pd.DataFrame(results, columns=["SMILES", "Predicted pIC50", "Bioactivity Class"])

def get_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">📥 Download CSV</a>'

# App Layout
def main():
    st.title("🧠 NeuroPlex")
    st.markdown("## NeuroPlex – An Innovative Neuro-AI Approach in Alzheimer's Therapeutics")

    st.markdown("""
    <div style='text-align: justify; font-size: 16px; line-height: 1.6; color: #000000;'>
        Welcome to <b>NeuroPlex</b>, a cutting-edge prediction platform designed to accelerate drug discovery for Alzheimer’s Disease.
        Powered by an advanced machine learning-based regression model, NeuroPlex delivers an outstanding <b>99% prediction accuracy</b> for pIC₅₀ values, 
        enabling researchers to evaluate the inhibitory potential of compounds with exceptional precision. <br><br>
        This intelligent tool deciphers complex molecular interactions and provides deep insights into compound bioactivity, 
        making it an invaluable asset in targeting key biomarkers associated with Alzheimer’s pathology. 
        Whether you're optimizing lead molecules or screening novel candidates, NeuroPlex empowers you to make data-driven decisions with confidence, 
        opening new avenues for therapeutic breakthroughs against Alzheimer’s Disease.
    </div>
    """, unsafe_allow_html=True)

    input_method = st.radio("Choose Input Method", ["Paste SMILES", "Upload File"])

    if input_method == "Paste SMILES":
        smiles_input = st.text_area("🧪 Enter SMILES strings (one per line)")
        if st.button("🔍 Predict"):
            smiles_list = [s.strip() for s in smiles_input.splitlines() if s.strip()]
            if not smiles_list:
                st.warning("⚠️ Please enter valid SMILES.")
            else:
                df = predict_pIC50_and_class(smiles_list)
                st.success("✅ Prediction complete!")
                st.dataframe(df)
                st.markdown(get_download_link(df), unsafe_allow_html=True)

    else:
        file = st.file_uploader("📤 Upload a CSV or TXT file with SMILES", type=["csv", "txt"])
        if file and st.button("🔍 Predict"):
            try:
                df = pd.read_csv(file, header=None)
                smiles_list = df.iloc[:, 0].dropna().astype(str).tolist()
                results = predict_pIC50_and_class(smiles_list)
                st.success("✅ Prediction complete!")
                st.dataframe(results)
                st.markdown(get_download_link(results), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Error: {e}")

    st.markdown("---")
    st.markdown("## 👨‍🔬 NeuroPlex Team")

    # Define columns for the profiles
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("""
            <div style='line-height: 1.3; color: #000000;'>
                <h3 style='color:#006a4e;'>Dr. Kashif Iqbal Sahibzada</h3>
                Assistant Professor<br>
                Department of Health Professional Technologies,<br>
                Faculty of Allied Health Sciences,<br>
                The University of Lahore<br>
                Post-Doctoral Fellow<br>
                Henan University of Technology, Zhengzhou, China<br>
                <b>Email:</b> kashif.iqbal@dhpt.uol.edu.pk | kashif.iqbal@haut.edu.cn
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div style='line-height: 1.3; color: #000000;'>
                <h3 style='color:#006a4e;'>Dr. Andleeb Batool</h3>
                Assistant Professor<br>
                Department of Zoology<br>
                Government College University, Lahore<br>
                <b>Email:</b> andleeb.batool@gcu.edu.pk
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div style='line-height: 1.3; color: #000000;'>
                <h3 style='color:#006a4e;'>Shumaila Shahid</h3>
                MS Biochemistry<br>
                School of Biochemistry and Biotechnology<br>
                University of the Punjab, Lahore<br>
                <b>Email:</b> shumaila.ms.sbb@pu.edu.pk
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
