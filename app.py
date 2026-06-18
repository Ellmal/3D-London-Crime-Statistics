import streamlit as st

st.set_page_config(
    page_title="London Crime Pulse Explorer",
    page_icon="🗺️",
    layout="wide",
)

st.title("London Crime Pulse Explorer")
st.caption("3D hexbin map of London street-level crime — work in progress")

st.info(
    "This is a placeholder app. Data loading, cleaning, and maps will be added "
    "in later minis. Testing starts with May 2025 (`2025-05`)."
)
