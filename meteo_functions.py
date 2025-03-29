
import pandas as pd
import requests
import numpy as np

lat = 49.13114
lon = 15.18067

def get_meteo_data(start_date, end_date):
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,relative_humidity_2m,surface_pressure,cloudcover,windspeed_10m,wind_direction_10m,direct_normal_irradiance,diffuse_radiation,shortwave_radiation&timezone=UTC"

    response = requests.get(url)
    if response.status_code == 200:
        weather_data = response.json()

        if "hourly" in weather_data:
            df_weather = pd.DataFrame({
                "DT": pd.to_datetime(weather_data["hourly"]["time"]),
                "Temperature_2m": weather_data["hourly"].get("temperature_2m", []),
                "Relative_Humidity_2m": weather_data["hourly"].get("relative_humidity_2m", []),
                "Surface_Pressure": weather_data["hourly"].get("surface_pressure", []),
                "Cloud_Cover": weather_data["hourly"].get("cloudcover", []),
                "Wind_Speed_10m": weather_data["hourly"].get("windspeed_10m", []),
                "Wind_Direction_10m": weather_data["hourly"].get("wind_direction_10m", []),
                "RAD": weather_data["hourly"].get("shortwave_radiation", [])
            })

            df_weather["DT"] = df_weather["DT"].dt.tz_localize(None)

            df_weather["wind_u"] = df_weather["Wind_Speed_10m"] * np.sin(np.radians(df_weather["Wind_Direction_10m"]))
            df_weather["wind_v"] = df_weather["Wind_Speed_10m"] * np.cos(np.radians(df_weather["Wind_Direction_10m"]))

            df_weather.drop(columns=["Wind_Speed_10m", "Wind_Direction_10m"], inplace=True)
        else:
            print("Chyba: Odpověď neobsahuje klíč 'hourly'.")

    else:
        print(f"Chyba při stahování dat: {response.status_code}, odpověď: {response.text}")

    return df_weather

def get_air_quality_data(start_date, end_date):

    url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=pm10,ozone&timezone=UTC"

    response = requests.get(url)
    if response.status_code == 200:
        air_quality_data = response.json()

        if "hourly" in air_quality_data:
            df_air_quality = pd.DataFrame({
                "DT": pd.to_datetime(air_quality_data["hourly"]["time"]),
                "PM10": air_quality_data["hourly"].get("pm10", []),
                "Ozone": air_quality_data["hourly"].get("ozone", [])
            })
            df_air_quality["DT"] = df_air_quality["DT"].dt.tz_localize(None)
        else:
            print("Chyba: Odpověď neobsahuje klíč 'hourly'.")
    else:
        print(f"Chyba při stahování dat: {response.status_code}, odpověď: {response.text}")
    
    return df_air_quality

def create_time_cycles(weather_dataset):
    weather_dataset["hour"] = weather_dataset["DT"].dt.hour
    weather_dataset["day_of_year"] = weather_dataset["DT"].dt.dayofyear
    weather_dataset["sin_hour"] = np.sin(2 * np.pi * weather_dataset["hour"] / 24)
    weather_dataset["cos_hour"] = np.cos(2 * np.pi * weather_dataset["hour"] / 24)
    weather_dataset["sin_day_of_year"] = np.sin(2 * np.pi * weather_dataset["day_of_year"] / 365)
    weather_dataset["cos_day_of_year"] = np.cos(2 * np.pi * weather_dataset["day_of_year"] / 365)
    weather_dataset.drop(columns=["hour", "day_of_year"], inplace=True)
    return weather_dataset

def get_forecast_meteo_data(start_date, end_date):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,relative_humidity_2m,surface_pressure,cloudcover,windspeed_10m,wind_direction_10m,direct_normal_irradiance,diffuse_radiation,shortwave_radiation&timezone=UTC"

    # Odeslání požadavku
    response = requests.get(url)
    if response.status_code == 200:
        weather_data = response.json()

        if "hourly" in weather_data:
            df_weather = pd.DataFrame({
                "DT": pd.to_datetime(weather_data["hourly"]["time"]),
                "Temperature_2m": weather_data["hourly"].get("temperature_2m", []),
                "Relative_Humidity_2m": weather_data["hourly"].get("relative_humidity_2m", []),
                "Surface_Pressure": weather_data["hourly"].get("surface_pressure", []),
                "Cloud_Cover": weather_data["hourly"].get("cloudcover", []),
                "Wind_Speed_10m": weather_data["hourly"].get("windspeed_10m", []),
                "Wind_Direction_10m": weather_data["hourly"].get("wind_direction_10m", []),
                "RAD": weather_data["hourly"].get("shortwave_radiation", [])
            })

            df_weather["DT"] = df_weather["DT"].dt.tz_localize(None)

            df_weather["wind_u"] = df_weather["Wind_Speed_10m"] * np.sin(np.radians(df_weather["Wind_Direction_10m"]))
            df_weather["wind_v"] = df_weather["Wind_Speed_10m"] * np.cos(np.radians(df_weather["Wind_Direction_10m"]))

            df_weather.drop(columns=["Wind_Speed_10m", "Wind_Direction_10m"], inplace=True)
        else:
            print("Chyba: Odpověď neobsahuje klíč 'hourly'.")
    else:
        print(f"Chyba při stahování dat: {response.status_code}, odpověď: {response.text}")

    return df_weather

def get_air_quality_forecast(start_date, end_date):
    url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=pm10,ozone&timezone=UTC"

    response = requests.get(url)
    if response.status_code == 200:
        air_quality_data = response.json()

        if "hourly" in air_quality_data:
            times = air_quality_data["hourly"].get("time", [])
            pm10 = air_quality_data["hourly"].get("pm10", [])
            ozone = air_quality_data["hourly"].get("ozone", [])

            if len(times) == len(pm10) == len(ozone):
                df_air_quality = pd.DataFrame({
                    "DT": pd.to_datetime(times),
                    "PM10": pm10,
                    "Ozone": ozone
                })
                df_air_quality["DT"] = df_air_quality["DT"].dt.tz_localize(None)  
            else:
                print("Chyba: Pola mají různé délky!")
                print(f"Počet časových údajů: {len(times)}, Počet PM10: {len(pm10)}, Počet Ozone: {len(ozone)}")
                df_air_quality = pd.DataFrame() 
        else:
            print("Chyba: Odpověď neobsahuje klíč 'hourly'.")
            df_air_quality = pd.DataFrame()  
    else:
        print(f"Chyba při stahování dat: {response.status_code}, odpověď: {response.text}")
        df_air_quality = pd.DataFrame()  
    
    return df_air_quality




    