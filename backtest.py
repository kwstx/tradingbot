import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from weather_arb_agent import CITY_COORDS, AgentState, DEFAULT_BANKROLL

# We will reuse the core logic but mock the external world
from weather_arb_agent import analyst_agent, decision_agent

class BacktestEngine:
    def __init__(self, historical_data_path, fee_pct=0.001, slippage_pct=0.002, train_split=0.7):
        """
        historical_data_path: Path to a JSON/CSV containing:
        - timestamp
        - city
        - market_price (YES token price)
        - actual_temp (for resolution)
        - forecast_mu, forecast_sigma (ensemble stats)
        
        fee_pct: Exchange commission (e.g., 0.1% = 0.001)
        slippage_pct: Average slippage per trade (e.g., 0.2% = 0.002)
        train_split: Fraction of data used for 'In-Sample' training/optimization.
        """
        self.data = pd.read_json(historical_data_path)
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.train_split = train_split
        self.results = []
        self.bankroll = DEFAULT_BANKROLL
        self.trades = []

    def split_data(self):
        """
        Splits data into In-Sample (Train) and Out-of-Sample (Test) sets.
        """
        split_idx = int(len(self.data) * self.train_split)
        train_data = self.data.iloc[:split_idx]
        test_data = self.data.iloc[split_idx:]
        return train_data, test_data

    def run(self, use_test_set=True):
        train_data, test_data = self.split_data()
        target_data = test_data if use_test_set else train_data
        
        mode_str = "OUT-OF-SAMPLE (Test)" if use_test_set else "IN-SAMPLE (Train)"
        print(f"Starting backtest in {mode_str} mode on {len(target_data)} data points...")
        
        for idx, row in target_data.iterrows():
            # Construct a mock state (v2.0 compatible)
            state: AgentState = {
                "current_markets": [{
                    "id": f"mkt_{idx}",
                    "question": f"Will {row['city']} temperature be above {row['threshold']}?",
                    "price_yes": row['market_price'],
                    "price_no": 1.0 - row['market_price'],
                    "city": row['city'],
                    "lat": row.get('lat', 40.7),
                    "lon": row.get('lon', -74.0),
                    "tz": "UTC"
                }],
                "weather_forecasts": {
                    f"mkt_{idx}": {
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
                "cycle_logs": [],
                "bankroll": self.bankroll,
                "api_weights": {"open_meteo": 1.0},
                "open_positions": []
            }

            # 1. Run Analyst
            analysis = analyst_agent(state)
            state.update(analysis)

            # 2. Run Decision
            decision = decision_agent(state)
            state.update(decision)

            # 3. Resolve Trade
            market_id = f"mkt_{idx}"
            if market_id in state["position_sizes"]:
                size = state["position_sizes"][market_id]
                side = state["trade_sides"][market_id]
                price_yes = row['market_price']
                threshold = row['threshold']
                actual_temp = row['actual_temp']
                
                # Slippage and Fee adjustments
                # If buying YES, entry price increases. If buying NO, entry price for NO increases.
                if side == "buy_yes":
                    entry_price = price_yes + (price_yes * self.slippage_pct)
                else:
                    price_no = 1.0 - price_yes
                    entry_price = price_no + (price_no * self.slippage_pct)
                
                # Cap entry price to avoid impossible scenarios
                entry_price = min(0.999, entry_price)
                
                # Fees are applied to the total capital invested
                fee = size * self.fee_pct
                
                # Did we win?
                condition_met = actual_temp > threshold
                won = (side == "buy_yes" and condition_met) or (side == "buy_no" and not condition_met)
                
                # PnL Calc: 
                # Number of tokens = size / entry_price
                # If win: PnL = (units * 1.0) - size - fee
                # If loss: PnL = -size - fee
                units = size / entry_price
                if won:
                    pnl = units - size - fee
                else:
                    pnl = -size - fee

                self.bankroll += pnl
                self.trades.append({
                    "timestamp": row['timestamp'],
                    "market": market_id,
                    "side": side,
                    "entry_price": entry_price,
                    "won": won,
                    "pnl": pnl,
                    "bankroll": self.bankroll,
                    "fee": fee
                })

        self.report(mode_str)

    def report(self, mode_str):
        df_trades = pd.DataFrame(self.trades)
        if df_trades.empty:
            print(f"No trades executed during {mode_str} backtest.")
            return

        roi = (self.bankroll - DEFAULT_BANKROLL) / DEFAULT_BANKROLL
        returns = df_trades['pnl'] / DEFAULT_BANKROLL
        sharpe = (returns.mean() / returns.std()) * np.sqrt(365) if len(returns) > 1 and returns.std() > 0 else 0
        
        print("\n" + "="*40)
        print(f"BACKTEST RESULTS: {mode_str}")
        print("="*40)
        print(f"Final Bankroll:   ${self.bankroll:.2f}")
        print(f"Total ROI:        {roi:.2%}")
        print(f"Sharpe Ratio:     {sharpe:.2f}")
        print(f"Total Trades:     {len(df_trades)}")
        print(f"Win Rate:         {(df_trades['won'].sum() / len(df_trades)):.2%}")
        print(f"Total Fees Paid:  ${df_trades['fee'].sum():.2f}")
        print("="*40)

if __name__ == "__main__":
    data_file = "actual_weather_data.json"
    import os
    if not os.path.exists(data_file):
        # Fallback to dummy data generation only if actual data is missing
        dummy_data = []
        for i in range(100):
            base_temp = 65 + (i * 0.1) 
            actual = base_temp + np.random.normal(0, 3)
            forecast_mu = base_temp + np.random.normal(0, 1.5)
            dummy_data.append({
                "timestamp": (datetime(2026, 1, 1) + timedelta(days=i)).isoformat(),
                "city": "NYC",
                "threshold": 70,
                "market_price": 0.4 + np.random.uniform(-0.1, 0.1),
                "actual_temp": actual,
                "forecast_mu": forecast_mu,
                "forecast_sigma": 2.0,
                "lat": 40.7128,
                "lon": -74.0060
            })
        with open(data_file, "w") as f:
            json.dump(dummy_data, f)
        
    # Initialize Engine with Fees and Slippage
    engine = BacktestEngine(data_file, fee_pct=0.002, slippage_pct=0.005, train_split=0.7)
    
    # 1. Run In-Sample (Optimization Phase)
    engine.run(use_test_set=False)
    
    # 2. Reset and Run Out-of-Sample (Validation Phase)
    engine.bankroll = DEFAULT_BANKROLL # Reset for clean test
    engine.trades = []
    engine.run(use_test_set=True)

