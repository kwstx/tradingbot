import json
import numpy as np
import pandas as pd
from datetime import datetime
from weather_arb_agent import CITY_COORDS, AgentState

# We will reuse the core logic but mock the external world
from weather_arb_agent import analyst_agent, decision_agent

class BacktestEngine:
    def __init__(self, historical_data_path):
        """
        historical_data_path: Path to a JSON/CSV containing:
        - timestamp
        - city
        - market_price (YES token price)
        - actual_temp (for resolution)
        - forecast_mu, forecast_sigma (ensemble stats)
        """
        self.data = pd.read_json(historical_data_path)
        self.results = []
        self.bankroll = 50.0
        self.trades = []

    def run(self):
        print(f"Starting backtest on {len(self.data)} data points...")
        
        for idx, row in self.data.iterrows():
            # Construct a mock state
            state: AgentState = {
                "current_markets": [{
                    "id": f"mkt_{idx}",
                    "question": f"Will {row['city']} temperature be above {row['threshold']}?",
                    "current_odds": row['market_price'],
                    "city": row['city']
                }],
                "weather_forecasts": {
                    row['city']: {
                        "open_meteo": {
                            "hourly": {
                                "temperature_2m": np.random.normal(row['forecast_mu'], row['forecast_sigma'], 24).tolist()
                            }
                        }
                    }
                },
                "probabilities": {},
                "edge_deltas": {},
                "ev_values": {},
                "position_sizes": {},
                "trade_sides": {},
                "risk_flags": [],
                "human_approval": True,
                "cycle_logs": []
            }

            # 1. Run Analyst
            analysis = analyst_agent(state)
            state.update(analysis)

            # 2. Run Decision
            decision = decision_agent(state)
            state.update(decision)

            # 3. Resolve Trade (Simple resolution)
            market_id = f"mkt_{idx}"
            if market_id in state["position_sizes"]:
                size = state["position_sizes"][market_id]
                side = state["trade_sides"][market_id]
                price = row['market_price']
                threshold = row['threshold']
                actual_temp = row['actual_temp']
                
                # Did we win?
                condition_met = actual_temp > threshold
                won = (side == "buy_yes" and condition_met) or (side == "buy_no" and not condition_met)
                
                # PnL Calc: If buy YES at 0.4, profit is 0.6 if win, -0.4 if loss
                if side == "buy_yes":
                    pnl = size * (1 - price) / price if won else -size
                else:
                    # If buy NO, we are effectively buying the YES token for (1-price)
                    # No, actually pmxt buy_no means buying the NO token. 
                    # Price of NO token = 1 - price_yes
                    # p_no = 1 - price
                    # pnl = size * (1 - p_no) / p_no if won else -size
                    p_no = 1 - price
                    pnl = size * (1 - p_no) / p_no if won else -size

                self.bankroll += pnl
                self.trades.append({
                    "timestamp": row['timestamp'],
                    "pnl": pnl,
                    "bankroll": self.bankroll
                })

        self.report()

    def report(self):
        df_trades = pd.DataFrame(self.trades)
        if df_trades.empty:
            print("No trades executed during backtest.")
            return

        roi = (self.bankroll - 50.0) / 50.0
        # Simple Sharpe (assuming daily trades)
        returns = df_trades['pnl'] / 50.0
        sharpe = (returns.mean() / returns.std()) * np.sqrt(365) if returns.std() > 0 else 0
        
        print("\n" + "="*30)
        print("BACKTEST RESULTS")
        print("="*30)
        print(f"Final Bankroll: ${self.bankroll:.2f}")
        print(f"Total ROI: {roi:.2%}")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Total Trades: {len(df_trades)}")
        print("="*30)

if __name__ == "__main__":
    # Create a dummy historical data for testing the backtester
    dummy_data = []
    for i in range(30):
        dummy_data.append({
            "timestamp": f"2026-03-{i+1:02d}T12:00:00",
            "city": "NYC",
            "threshold": 65,
            "market_price": 0.4 + np.random.uniform(-0.1, 0.1),
            "actual_temp": 68 + np.random.uniform(-5, 5),
            "forecast_mu": 67,
            "forecast_sigma": 2.0
        })
    
    with open("dummy_hist_data.json", "w") as f:
        json.dump(dummy_data, f)
        
    engine = BacktestEngine("dummy_hist_data.json")
    engine.run()
