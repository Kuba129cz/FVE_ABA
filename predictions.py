import streamlit as st
import pandas as pd
import numpy as np
from meteo_functions import create_time_cycles
import joblib
from preprocessing_functions import create_sequences, plot_solar_power_prediction, load_model
from tensorflow import keras

def predict(data):
    data = create_time_cycles(data)
    st.divider()
    # Načtění transformátorů
    input_preprocessor = joblib.load('input_preprocessor_meteo_to_smape25.pkl')
    output_scaler = joblib.load('output_scaler_meteo_to_smape25.pkl')

    # transformace vstupnich prom
    X_dataset = input_preprocessor.transform(data)
    X_dataset = pd.DataFrame(X_dataset, columns=input_preprocessor.get_feature_names_out(), index=data.index)

    rename_map = {
        "yeo_minmax__RAD": "RAD",
        "yeo_minmax__Relative_Humidity_2m": "Relative_Humidity_2m",
        "yeo_minmax__PM10": "PM10",
        "yeo_standard__Cloud_Cover": "Cloud_Cover",
        "minmax__Temperature_2m": "Temperature_2m",
        "minmax__wind_u": "wind_u",
        "minmax__wind_v": "wind_v",
        "minmax__Surface_Pressure": "Surface_Pressure",
        "minmax__Ozone": "Ozone",
        "remainder__sin_hour": "sin_hour",
        "remainder__cos_hour": "cos_hour",
        "remainder__sin_day_of_year": "sin_day_of_year",
        "remainder__cos_day_of_year": "cos_day_of_year"
    }
    X_dataset = X_dataset.rename(columns=rename_map)

    features = ["RAD", "Relative_Humidity_2m", "Surface_Pressure", 'Cloud_Cover', "Temperature_2m",
            'sin_hour', 'cos_hour', 'cos_day_of_year', "sin_day_of_year", "wind_u", "wind_v", "Ozone", "PM10"]

    x = create_sequences(X_dataset, window=24, horizon=24, past_features=features, future_features=features)

    model = load_model() 

    y_pred = model.predict(x) # predikce

    y_pred_trans = np.array([output_scaler.inverse_transform(y_pred[:, i].reshape(-1, 1)).flatten() for i in range(y_pred.shape[1])]).T

    y_pred_trans[y_pred_trans < 1] = 0
    st.write("**Predikované hodnoty výkonu (kW) pro jednotlivé hodiny:**")
    st.write(y_pred_trans)

    my_plt = plot_solar_power_prediction(y_pred_trans)
    st.pyplot(my_plt)

    return None
