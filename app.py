# app.py
import streamlit as st
import pandas as pd
import numpy as np
import base64
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
    # Set the color scheme
    header_color = '#91C788'
    background_color = '#FFFFFF'
    text_color = '#333333'
    primary_color = '#800000'
    footer_color = '#017C8C'
    footer_text_color = '#FFFFFF'
    font = 'Arial, sans serif'

    # Set the page config
    st.set_page_config(
        page_title='OctaScanner',
        layout='wide',
        initial_sidebar_state='expanded',
        page_icon='🎡',
    )

    # Set the theme
    st.markdown(f"""
    <style>
        .reportview-container {{
            background-color: {background_color};
            color: {text_color};
            font-family: {font};
        }}
        .sidebar .sidebar-content {{
            background-color: {header_color};
            color: {text_color};
        }}
        .stButton > button {{
            background-color: {primary_color};
            color: {background_color};
            border-radius: 12px;
            font-size: 16px;
            padding: 10px 20px;
        }}
        footer {{
            font-family: {font};
            background-color: {footer_color};
            color: {footer_text_color};
        }}
        .header-title {{
            color: {primary_color};
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
        }}
        .header-subtitle {{
            color: {text_color};
            font-size: 20px;
            text-align: center;
            margin-bottom: 30px;
        }}
    </style>
    """, unsafe_allow_html=True)
     # Add the image and title at the top of the page
    col1, col2, col3 = st.columns([1,2,3])
    with col1:
        st.image("hiv2.jpg", width=580)
    with col3:
        st.markdown("<h1 class='header-title'>NeuroPlex – An Innovative Approach towards Alzheimer Therapeutics</h1>", unsafe_allow_html=True)
        st.markdown("""
        <p class='header-subtitle'>
       Welcome to NeuroPlex, a cutting-edge prediction platform designed to accelerate drug discovery for Alzheimer’s Disease. Powered by an advanced machine learning-based regression model, NeuroPlex delivers an outstanding 99% prediction accuracy for pIC₅₀ values, enabling researchers to evaluate the inhibitory potential of compounds with exceptional precision. This intelligent tool deciphers complex molecular interactions and provides deep insights into compound bioactivity, making it an invaluable asset in targeting key biomarkers associated with Alzheimer’s pathology. Whether you're optimizing lead molecules or screening novel candidates, NeuroPlex empowers you to make data-driven decisions with confidence, opening new avenues for therapeutic breakthroughs against Alzheimer’s Disease.
        </p>
        """, unsafe_allow_html=True)
# Add university logos to the page
    left_logo, center, right_logo = st.columns([1, 2, 1])
    center.image("ref.jpg", width=650)
    #right_logo.image("image.jpg", width=250)
if __name__ == "__main__":
    main()
# Load model that was saved with only standard scikit-learn types
model = joblib.load("model.pkl")

st.set_page_config(page_title="NeuroPlex", layout="wide")

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
                "Active" if pIC50 >= 6 else
                "Intermediate" if pIC50 >= 5 else
                "Inactive"
            )
            results.append((smi, round(pIC50, 2), activity))
        else:
            results.append((smi, None, "Invalid SMILES"))
    return pd.DataFrame(results, columns=["SMILES", "Predicted pIC50", "Bioactivity Class"])

def get_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">📥 Download CSV</a>'

def main():
    st.title("🧠 NeuroPlex")
    st.markdown("AI-powered prediction of **pIC₅₀** and bioactivity class for Alzheimer's drug candidates.")

    input_method = st.radio("Choose Input Method", ["Paste SMILES", "Upload File"])

    if input_method == "Paste SMILES":
        smiles_input = st.text_area("Enter SMILES strings (one per line)")
        if st.button("Predict"):
            smiles_list = [s.strip() for s in smiles_input.splitlines() if s.strip()]
            if not smiles_list:
                st.warning("Please enter valid SMILES.")
            else:
                df = predict_pIC50_and_class(smiles_list)
                st.success("✅ Prediction complete!")
                st.dataframe(df)
                st.markdown(get_download_link(df), unsafe_allow_html=True)

    else:  # Upload File
        file = st.file_uploader("Upload a CSV or TXT file with SMILES", type=["csv", "txt"])
        if file and st.button("Predict"):
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
    st.markdown("#### 👨‍🔬 Team NeuroPlex")
    st.markdown("""
    - **Dr. Kashif Iqbal Sahibzada** – UOL & HAUT  
    - **Dr. Andleeb Batool** – GCU Lahore  
    - **Shumaila Shahid** – PU Lahore
    """)

if __name__ == "__main__":
    main()

