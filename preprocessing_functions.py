import numpy as np
import tensorflow as tf
import streamlit as st
import os
import requests
import matplotlib.pyplot as plt

MODEL_URL = "https://github.com/Kuba129cz/FVE_ABA/releases/download/model_FVE/output_predictions_to_meteo_smape_25.keras"
MODEL_PATH = "model.keras"

def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Stahuji model, prosím čekejte...")
        response = requests.get(MODEL_URL, stream=True)
        
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success("Model byl úspěšně stažen.")
        else:
            st.error("Nepodařilo se stáhnout model. Zkontrolujte URL.")
    else:
        st.info("Model již je stažen.")

@st.cache_resource  # To zajistí, že model je načten a uložen v cache mezi jednotlivými relacemi
def load_model():
    download_model()  # Zavoláme download_model pouze jednou, pokud není model k dispozici
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

def create_sequences(data, window, horizon, past_features, future_features):
    """
    Vytvoří sekvence vstupních dat a odpovídající cílové hodnoty pro trénování LSTM modelu.

    Parametry:
    ----------
    data : pandas.DataFrame
        DataFrame obsahující časové řady.
    window : int
        Počet časových kroků v minulosti.
    horizon : int
        Počet časových kroků do budoucnosti.
    past_features : list
        Seznam sloupců, které budou použity jako vstupní vlastnosti v minulosti.
    future_features : list
        Seznam sloupců, které budou použity jako vstupní vlastnosti v budoucnosti.
    target : str
        Název sloupce, který bude použit jako cílová hodnota.

    Návratové hodnoty:
    -------------------
    X : numpy.ndarray
        Pole tvaru (vzorky, window, past_features + future_features), obsahující sekvence vstupních dat.
    y : numpy.ndarray
        Pole tvaru (vzorky, horizon), obsahující odpovídající cílové hodnoty.
    """

    X_past = np.lib.stride_tricks.sliding_window_view(
        data[past_features].values, (window, len(past_features))
    )[:-horizon, :, :]

    X_past = np.squeeze(X_past, axis=1)  

    X_future = np.lib.stride_tricks.sliding_window_view(
        data[future_features].values, (window, len(future_features))
    )[horizon-1 : len(X_past) + horizon-1, :, :]

    X_future = np.squeeze(X_future, axis=1)  

    X = np.concatenate([X_past, X_future], axis=2)  

    return X

def plot_solar_power_prediction(y_pred_trans):
    """
    Vytvoří graf predikce výkonu fotovoltaické elektrárny v průběhu dne.
    
    Args:
        y_pred_trans (numpy.ndarray): Pole s predikovanými hodnotami výkonu (kW) ve tvaru (1, 24).
    
    Returns:
        plt.Figure: Graf pro zobrazení.
    """
    hours = np.arange(24)
    
    plt.figure(figsize=(10, 6))
    plt.plot(hours, y_pred_trans.flatten(), marker='o', label='Výkon (kW)')
    plt.xlabel('Hodiny')
    plt.ylabel('Výkon (kW)')
    plt.title('Predikce výkonu fotovoltaické elektrárny v průběhu dne')
    plt.xticks(hours) 
    plt.grid(True)
    plt.legend()
    
    return plt

