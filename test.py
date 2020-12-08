import streamlit as st
import pandas as pd
from pathlib import Path

uploaded_file = st.file_uploader("Choose a XLSX file", type=["xlsx","csv"])

if uploaded_file:
    extension = Path(uploaded_file.name).suffix
    if extension.upper()==".XLSX":

        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    st.dataframe(df)


