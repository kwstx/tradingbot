import httpx
import time
from datetime import datetime, timedelta
from persistence import PersistenceManager

db = PersistenceManager()

class ReliabilityManager:
    """
    Backtest Loop Service:
    Compares past forecasts against actual resolution data and updates API weights.
    """
    
    @staticmethod
    def fetch_actual_weather(lat: float, lon: float, date_str: str) -> float:
        """
        Fetches historical temperature for a specific date and location from Open-Meteo.
        date_str format: YYYY-MM-DD
        """
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={date_str}&end_date={date_str}&hourly=temperature_2m"
        try:
            with httpx.Client() as client:
                resp = client.get(url, timeout=10.0)
                if resp.status_code == 200:
                    data = resp.json()
                    temps = data.get("hourly", {}).get("temperature_2m", [])
                    if temps:
                        # Return the max temperature of the day as a common resolution metric
                        # (Adjust this based on the specific market resolution rules if needed)
                        return max(temps)
        except Exception as e:
            print(f"[RELIABILITY] Failed to fetch historical weather: {e}")
        return None

    @classmethod
    def run_backtest_loop(cls):
        """
        Identifies resolved markets and updates reliability weights.
        """
        print("\n[RELIABILITY] Starting Historical Forecast Backtest Loop...")
        unresolved = db.get_unresolved_forecasts()
        
        if not unresolved:
            print(" -> No unresolved forecasts found.")
            return

        print(f" -> Found {len(unresolved)} forecasts awaiting resolution.")
        resolved_count = 0
        
        for f_id, m_id, api_name, target_date, mu, lat, lon in unresolved:
            # Check if target_date has passed (we need at least 1 day after to have historical data)
            target_dt = datetime.strptime(target_date, "%Y-%m-%d")
            if target_dt >= datetime.now() - timedelta(days=1):
                continue
            
            actual = cls.fetch_actual_weather(lat, lon, target_date)
            if actual is not None:
                db.resolve_forecast(f_id, actual)
                print(f" -> Resolved {m_id} ({api_name}): Forecast {mu:.1f} | Actual {actual:.1f} | Error {abs(mu-actual):.1f}")
                resolved_count += 1
            
            # Rate limiting
            time.sleep(1)

        print(f"[RELIABILITY] Backtest cycle complete. Resolved {resolved_count} forecasts.")

if __name__ == "__main__":
    ReliabilityManager.run_backtest_loop()
