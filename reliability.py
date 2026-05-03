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
        
        # 1. Resolve Paper Trades FIRST (High Priority)
        cls.resolve_paper_trades()
        
        # 2. Process Historical Backlog (Limited per cycle to prevent blocking)
        unresolved = db.get_unresolved_forecasts()
        
        if not unresolved:
            print(" -> No unresolved forecasts found.")
            return

        print(f" -> Found {len(unresolved)} forecasts awaiting resolution.")
        resolved_count = 0
        
        for f_id, m_id, api_name, target_date, mu, lat, lon in unresolved[:200]: # Limit to 200 per cycle
            # Check if target_date has passed (we need historical data availability)
            target_dt = datetime.strptime(target_date, "%Y-%m-%d")
            # If target date was yesterday or older, we can usually resolve
            if target_dt >= datetime.now().replace(hour=0, minute=0, second=0, microsecond=0):
                continue
            
            actual = cls.fetch_actual_weather(lat, lon, target_date)
            if actual is not None:
                db.resolve_forecast(f_id, actual)
                print(f" -> Resolved {m_id} ({api_name}): Forecast {mu:.1f} | Actual {actual:.1f} | Error {abs(mu-actual):.1f}")
                resolved_count += 1
            
            # Rate limiting
            time.sleep(0.1) # Reduced sleep for performance

        print(f"[RELIABILITY] Backtest cycle complete. Resolved {resolved_count} forecasts.")

    @classmethod
    def resolve_paper_trades(cls):
        """
        Processes paper trades, calculates PnL from resolved forecasts, and updates balance.
        """
        print("[RELIABILITY] Resolving Paper Trades...")
        unresolved_trades = db.get_unresolved_paper_trades()
        print(f" -> Found {len(unresolved_trades)} unresolved paper trades.")
        
        for t_id, m_id, side, size, price, target_date, threshold, lat, lon in unresolved_trades:
            target_dt = datetime.strptime(target_date, "%Y-%m-%d")
            # print(f" -> Checking Trade {t_id} (Target: {target_date})...")
            if target_dt >= datetime.now().replace(hour=0, minute=0, second=0, microsecond=0):
                continue
                
            actual = cls.fetch_actual_weather(lat, lon, target_date)
            if actual is not None and threshold != float('inf') and threshold != float('-inf'):
                # Logic: Win if side matches the outcome
                is_above = actual > threshold
                is_yes = "yes" in side.lower()
                won = (is_yes and is_above) or (not is_yes and not is_above)
                
                # PnL Calculation: 
                # tokens = size / price
                # if won: payout = tokens * 1.0; net_pnl = payout - size
                # if lost: net_pnl = -size
                tokens = size / price
                if won:
                    pnl = tokens - size
                else:
                    pnl = -size
                
                db.resolve_paper_trade(t_id, size, won, pnl)
                res_str = "WON" if won else "LOST"
                print(f" -> Paper Trade {t_id} Resolved: {res_str} | PnL: ${pnl:+.2f}")

if __name__ == "__main__":
    ReliabilityManager.run_backtest_loop()
