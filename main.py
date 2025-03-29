import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import datetime
from meteo_functions import get_meteo_data, get_air_quality_data, get_forecast_meteo_data, get_air_quality_forecast
from predictions import predict 
import gc

def clear_memory():
    gc.collect() 

st.title("Predikce výkonu FVE ABA")
#t.header("this is a header")
#sst.subheader("subheader")
#st.markdown("This is **Markdown**")
#st.caption("small text")

today = datetime.date.today()  
max_date = today + datetime.timedelta(days=4)

with st.form(key="sample_form"):
    date_utc = st.date_input("Vyber den", max_value=max_date)
    submit_button = st.form_submit_button(label="Predikuj")

if submit_button:   
    previous_day = date_utc - timedelta(days=1) 
    if date_utc < today - datetime.timedelta(days=1):
       # st.subheader(f"Predikce výkonu pro: {date_utc}:")

        df_meteo = get_meteo_data(previous_day, date_utc)
        df_air_quality = get_air_quality_data(previous_day, date_utc)

        data = df_meteo.merge(df_air_quality, on="DT", how="inner")
        predict(data)
        clear_memory()

    elif previous_day < date_utc <= today + datetime.timedelta(days=4):
       # st.header(f"Data pro: {date_utc}:")
        df_meteo = get_forecast_meteo_data(previous_day, date_utc)
        df_air_quality = get_air_quality_forecast(previous_day, date_utc)
        data = df_meteo.merge(df_air_quality, on="DT", how="inner")
        #st.write("Budouci data:")
        #st.dataframe(df_meteo)
        #st.dataframe(df_air_quality)
        #st.dataframe(data)
        predict(data)
        clear_memory()

    else:
        st.warning("Predikce je dostupná pouze pro následujících 5 dnů.")
