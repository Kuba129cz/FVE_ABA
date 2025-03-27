import streamlit as st
import pandas as pd
import numpy as np
import requests

st.title("Predikce výkonu FVE ABA")
#t.header("this is a header")
#sst.subheader("subheader")
#st.markdown("This is **Markdown**")
#st.caption("small text")
#st.divider()

## CHARTS
with st.form(key="sample_form"):
    date_utc = st.date_input("Vyber den")
    st.form_submit_button(label="klikni")
st.title(date_utc)

