# app.py
import streamlit as st
import pandas as pd
import numpy as np
import base64
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
# Set the page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title='NeuroPlex',
    layout='wide',
    initial_sidebar_state='expanded',
    page_icon='֎',
)

def main():
    # Set the color scheme
    header_color = '#E6E6FA'         # Maroon
    background_color = '#FFFFFF'     # White
    text_color = '#333333'           # Dark Gray
    primary_color = '#7A4E9F'        # Darker Maroon
    footer_color = '#6A4C9C'         # Deep Maroon
    footer_text_color = '#FFFFFF'    # White
    font = 'Arial, sans-serif'

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
 
 # Add header with application title and description
with st.container():  # Corrected from 'center' to 'st.container'
    st.markdown(
        "<h1 class='header-title'>NeuroPlex – An Artificial Intelligence Approach towards the Drug Discovery based on pIC50 value for Alzheimer's Disease</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <p class='header-subtitle'>
       Welcome to NeuroPlex, a powerful prediction server designed to assess the pIC50 values of compounds targeting therapeutically to Alzheimer's Disease. Built on a highly accurate machine learning-based regression model, NeuroPlex achieves an impressive 99% accuracy, enabling precise and reliable predictions. This tool deciphers complex molecular interactions, providing insights into the inhibitory potential of compounds to biomarkers. Join us in advancing drug discovery, unlocking novel therapeutic possibilities against Alzheimer's disease.
         </p>
        """,
        unsafe_allow_html=True
    )
    #st.image("erm.jpg", width=800)
    col1, col2, col3 = st.columns([1,2,3])
    with col2:
        st.image("erm.jpg", width=600)
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

if __name__ == "__main__":
    main()
# HTML and CSS to color the title and header
st.markdown(
    """
    <style>
    .title {
        color: #7A4E9F;  /* Parrot Green color code */
        font-size: 2em;
        font-weight: bold;
    }
    .header {
        color: #7A4E9F;  /* Parrot Green color code */
        font-size: 1.5em;
        font-weight: bold;
    }
    </style>
    <h1 class="title">Team NeuroPlex:</h1>
    """,
    unsafe_allow_html=True
)
 
# Define columns for the profiles
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    # st.image("my-photo.jpg", width=100)
    st.markdown("""
        <div style='line-height: 1.1;'>
            <h3>Dr. Kashif Iqbal Sahibzada</h3>
             Assistant Professor | Department of Health Professional Technologies, Faculty of Allied Health Sciences, The University of Lahore<br>
            Post-Doctoral Fellow | Henan University of Technology,Zhengzhou China<br>
            Email: kashif.iqbal@dhpt.uol.edu.pk | kashif.iqbal@haut.edu.cn
        </div>
    """, unsafe_allow_html=True)

with col2:
    # st.image("colleague-photo.jpg", width=100)
    st.markdown("""
        <div style='line-height: 1.1;'>
            <h3>Dr. Andleeb Batool</h3>
            Assistant Professor | Department of Zoology<br>
            Government College University, Lahore<br>
            Email: andleeb.batool@gcu.edu.pk
        </div>
    """, unsafe_allow_html=True)

with col3:
    # st.image("teacher-photo.jpg", width=100)
    st.markdown("""
        <div style='line-height: 1.1;'>
            <h3>Shumaila Shahid</h3>
            MS Biochemistry<br>
            School of Biochemistry and Biotechnology<br>
            University of the Punjab, Lahore<br>
            Email: shumaila.ms.sbb@pu.edu.pk
        </div>
    """, unsafe_allow_html=True)

#Add University Logo
left_logo, center_left, center_right, right_logo = st.columns([1, 1, 1, 1])
#left_logo.image("LOGO_u.jpeg", width=200)
center_left.image("uol.jpg", width=450)  # Replace with your center-left logo image
#right_logo.image("image.jpg", width=200) 
