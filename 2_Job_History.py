import streamlit as st
import pandas as pd

st.set_page_config(page_title="Job History", page_icon="🕒", layout="wide")

st.title("🕒 Job History (Current Session)")
st.markdown("This page shows a history of the files processed since you opened the app. The history will reset if you close the browser tab.")

# Check if the history list in session state is not empty
if 'history' in st.session_state and st.session_state.history:
    # Convert the history list to a DataFrame and sort by time
    history_df = pd.DataFrame(st.session_state.history).sort_values(by="time", ascending=False)
    st.dataframe(history_df, use_container_width=True, hide_index=True)
else:
    st.info("No files have been processed in this session yet. Go to the 'Process Files' page to begin.")