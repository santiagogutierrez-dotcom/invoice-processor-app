import streamlit as st
import pandas as pd
from lookup_data import lookup_dict

st.set_page_config(page_title="Lookup Data", page_icon="📖", layout="wide")

st.title("📖 Lookup Data")
st.markdown("This table contains the mapping from employee names to their respective cost centers (`Kostenstellen`).")

# Convert the list of dictionaries to a pandas DataFrame for better display
df = pd.DataFrame(lookup_dict)
st.dataframe(df, use_container_width=True)