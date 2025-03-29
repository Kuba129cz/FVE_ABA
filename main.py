import psutil
import streamlit as st

mem_info = psutil.virtual_memory()
st.write(f"Celková paměť: {mem_info.total / (1024 ** 3):.2f} GB")
st.write(f"Volná paměť: {mem_info.available / (1024 ** 3):.2f} GB")
