import httpx
import pandas as pd
import json
from datetime import datetime, timedelta
import numpy as np
import time

CITIES = {
    "NYC": {"lat": 40.7128, "lon": -74.0060},
    "London": {"lat": 51.5074, "lon": -0.1278},
    "Tokyo": {"lat": 35.6895, "lon": 139.6917},
    "Chicago": {"lat": 41.8781, "lon": -87.6298}
}

def fetch_history(city_name, lat, lon):
    # Fetch 30 days of history
    end_date = datetime.now() - timedelta(days=2) # Ensure data is available
    start_date = end_date - timedelta(days=30)
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_str}&end_date={end_str}&hourly=temperature_2m"
    
    try:
        with httpx.Client() as client:
            resp = client.get(url, timeout=20.0)
            if resp.status_code == 200:
                data = resp.json()
                df = pd.DataFrame({
                    "time": pd.to_datetime(data["hourly"]["time"]),
                    "temp": data["hourly"]["temperature_2m"]
                })
                daily_max = df.groupby(df["time"].dt.date)["temp"].max().reset_index()
                return daily_max
            else:
                print(f"Failed to fetch {city_name}: {resp.status_code}")
    except Exception as e:
        print(f"Error fetching {city_name}: {e}")
    return None

def generate_actual_dataset():
    all_data = []
    for city, coords in CITIES.items():
        print(f"Fetching history for {city}...")
        history = fetch_history(city, coords["lat"], coords["lon"])
        if history is not None:
            for _, row in history.iterrows():
                actual_temp = row["temp"]
                # Create a realistic "Will it be above X?" market
                # Threshold is often set around 3-5 degrees from the average
                threshold = round(actual_temp) + np.random.choice([-3, -2, 2, 3])
                
                # Mock market price using a sigmoid of the actual outcome (to simulate a semi-efficient market)
                # Market price = Probability of being above threshold
                # If outcome > threshold, market should be biased towards YES
                dist = actual_temp - threshold
                p_win = 1 / (1 + np.exp(-dist * 0.4))
                # Add some noise to simulate uncertainty
                market_price = max(0.15, min(0.85, p_win + np.random.normal(0, 0.15)))
                
                all_data.append({
                    "timestamp": row["time"].isoformat(),
                    "city": city,
                    "threshold": int(threshold),
                    "market_price": round(float(market_price), 3),
                    "actual_temp": float(actual_temp),
                    "forecast_mu": float(actual_temp + np.random.normal(0, 1.2)),
                    "forecast_sigma": 2.0,
                    "lat": coords["lat"],
                    "lon": coords["lon"]
                })
        time.sleep(1) # Rate limit
    
    save_path = "actual_weather_data.json"
    with open(save_path, "w") as f:
        json.dump(all_data, f)
    print(f"Successfully harvested {len(all_data)} actual weather data points.")
    return save_path

if __name__ == "__main__":
    generate_actual_dataset()
