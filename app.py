

import streamlit as st
import requests
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import AllChem, rdMolAlign
from rdkit.Chem import Descriptors, Lipinski
from rdkit import DataStructs
from rdkit.Chem import RDKFingerprint
from rdkit.Chem import PandasTools
from Bio import SeqIO
import joblib
from joblib import dump, load
import pickle 
import sklearn
from sklearn import svm
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import rcsbsearchapi
import os

def main():
    # Set the color scheme
    primary_color = '#4863A0'
    secondary_color = '#4169E1'
    tertiary_color = '#368BC1'
    background_color = '#F5F5F5'
    text_color = '#004225'
    font = 'sans serif'

    # Set the page config
    st.set_page_config(
        page_title='Drug Predictor Pro',
        layout= 'wide',
        initial_sidebar_state='expanded'
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
            background-color: {secondary_color};
            color: {tertiary_color};
        }}
        .streamlit-button {{
            background-color: {primary_color};
            color: {tertiary_color};
        }}
        footer {{
            font-family: {font};
        }}
    </style>
    """, unsafe_allow_html=True)
    
    
    # Add university logos to the page
    #left_logo, center, right_logo = st.columns([1, 2, 1])
    #left_logo.image("PU.png", width=280)
    #right_logo.image("LOGO_u.png", width=280)

    # Add header with application title and description
    with center:
      st.markdown("<h1 style='font-family:Bodoni MT Black;font-size:40px;'>Drug Predictor Pro</h1>", unsafe_allow_html=True)
      st.write("")
      st.markdown("<p style='font-family:Bodoni MT;font-size:20px;font-style: italic;'>Unlock the power of predictive analytics with Drug Predictor Pro, your cutting edge machine learning app designed to revolutionize healthcare by forecasting optimal drug responses for personalized treatment plans.</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()  


def calculate_lipinski(smiles_list):
    data = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            row = {
                "SMILES": smiles,
                "MW": Descriptors.MolWt(mol),
                "LogP": Descriptors.MolLogP(mol),
                "NumHDonors": Descriptors.NumHDonors(mol),
                "NumHAcceptors": Descriptors.NumHAcceptors(mol)
            }
            data.append(row)
        else:
            st.warning(f"Invalid SMILES: {smiles}")
    return pd.DataFrame(data)

def main():
    st.title("Drug Predictor Pro")
    st.header("Lipinski Descriptors Calculator")
    
    # Input widgets
    option = st.radio("Select input method:", ("Enter SMILES", "Upload SMILES from CSV"), key="input_method")
    if option == "Enter SMILES":
        smiles_input = st.text_area("Enter SMILES notation (separate by newline)", key="smiles_input")
        smiles_list = smiles_input.split('\n')
    else:
        smiles_csv_file = st.file_uploader("Upload smiles.csv file", type=["csv"], key="file_uploader")
        if smiles_csv_file is not None:
            df = pd.read_csv(smiles_csv_file)
            if 'SMILES' in df.columns:
                smiles_list = df['SMILES'].tolist()
            else:
                st.error("SMILES column not found in the uploaded CSV file.")
                return
        else:
            st.warning("Please upload a CSV file.")

    calculate_button = st.button("Calculate Lipinski Descriptors", key="calculate_button")
    if calculate_button:
        if not smiles_list:
            st.warning("Please enter SMILES or upload a CSV file.")
            return
        lipinski_descriptors = calculate_lipinski(smiles_list)
        st.write("Lipinski Descriptors:")
        st.write(lipinski_descriptors)

if __name__ == "__main__":
    main()

# Function to calculate IC50 value
def calculate_ic50(smiles):
    # Load the machine learning model
    model_file = "model2.pkl"
    model = joblib.load(model_file)

    # Calculate molecular descriptors using RDKit
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid SMILES"
    
    features = []
    for descriptor in Descriptors.descList:
        try:
            features.append(descriptor[1](mol))
        except:
            features.append(np.nan)

    # Predict pIC50 value
    pIC50_prediction = model.predict([features])[0]

    # Inverse transformation to obtain IC50 value
    ic50_value = 10 ** (-pIC50_prediction)

    return ic50_value

# Streamlit UI
st.header('IC50 Value Calculator')

# Input SMILES string
smiles_input = st.text_area("Enter SMILES notation (separate by newline)", key="smiles_input_ic50")
smiles_list = smiles_input.split('\n')

# Calculate IC50 button
if st.button("Calculate IC50"):
    if smiles_input:
        ic50_value = calculate_ic50(smiles_input)
        st.write(f"Predicted IC50 value: {ic50_value}")
    else:
        st.write("Please enter a valid SMILES string.")
