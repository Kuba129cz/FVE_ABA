import os
import requests
import tensorflow as tf
import streamlit as st

# URL modelu v GitHub Releases
MODEL_URL = "https://github.com/Kuba129cz/FVE_ABA/releases/download/model_FVE/output_predictions_to_meteo_smape_25.keras"
MODEL_PATH = "model.keras"

# Funkce pro stažení modelu, pokud neexistuje
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Stahuji model, prosím čekejte...")
        response = requests.get(MODEL_URL, stream=True)
        
        # Ověření úspěšného stažení
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success("Model byl úspěšně stažen.")
        else:
            st.error("Nepodařilo se stáhnout model. Zkontrolujte URL.")

# Streamlit cache pro rychlejší načítání modelu
@st.cache_resource
def load_model():
    download_model()  # Nejprve ověř, že model existuje
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# Načtení modelu
model = load_model()
st.success("Model byl úspěšně načten!")

# Otestování načtení modelu
st.write("Struktura modelu:")
model.summary(print_fn=lambda x: st.text(x))