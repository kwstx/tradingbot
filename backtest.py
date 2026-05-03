import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from weather_arb_agent import CITY_COORDS, AgentState, DEFAULT_BANKROLL

# We will reuse the core logic but mock the external world
from weather_arb_agent import analyst_agent, decision_agent

class ExecutionSimulator:
    """
    Simulates real-world execution hurdles:
    - Bid/Ask Spread
    - Partial Fills
    - Missed Orders
    - Latency
    - Fees
    """
    def __init__(self, fee_pct=0.001, fixed_fee=0.0, base_spread=0.01, latency_periods=1):
        self.fee_pct = fee_pct
        self.fixed_fee = fixed_fee
        self.base_spread = base_spread
        self.latency_periods = latency_periods

    def get_market_prices(self, mid_price):
        """Returns (bid, ask) based on mid price and spread."""
        # Spread is often wider at extremes (near 0 or 1)
        spread = self.base_spread * (1.0 + (1.0 - 4.0 * (mid_price - 0.5)**2)) 
        ask = min(0.999, mid_price + spread / 2)
        bid = max(0.001, mid_price - spread / 2)
        return bid, ask

    def simulate_execution(self, requested_size, side, current_idx, data):
        """
        Simulates the execution of an order.
        Returns (filled_size, execution_price, fee_paid)
        """
        # 1. Latency: Price might have moved by the time we execute
        exec_idx = min(len(data) - 1, current_idx + self.latency_periods)
        actual_row = data.iloc[exec_idx]
        mid_price = actual_row['market_price']
        bid, ask = self.get_market_prices(mid_price)
        
        # 2. Determine execution price based on side
        if side == "buy_yes":
            exec_price = ask
        elif side == "buy_no":
            # Market price is for YES, so NO price is 1 - YES price
            # But we buy NO at the 'ask' of NO, which is 1 - 'bid' of YES
            exec_price = 1.0 - bid
        else:
            return 0, 0, 0 # Should not happen in this simplified model

        # 3. Partial Fills & Missed Orders
        # Probability of being missed entirely (e.g. 5%)
        if np.random.random() < 0.05:
            return 0, exec_price, 0
        
        # Fill percentage (random between 40% and 100% for realistic partial fills)
        fill_pct = np.random.uniform(0.4, 1.0)
        filled_size = requested_size * fill_pct
        
        # 4. Fees
        fee = (filled_size * self.fee_pct) + self.fixed_fee
        
        return filled_size, exec_price, fee

class BacktestEngine:
    def __init__(self, historical_data_path, fee_pct=0.001, slippage_pct=0.002, train_split=0.7):
        self.data = pd.read_json(historical_data_path)
        self.train_split = train_split
        self.results = []
        self.bankroll = DEFAULT_BANKROLL
        self.trades = []
        
        # Initialize the new Execution Simulator
        self.simulator = ExecutionSimulator(
            fee_pct=fee_pct, 
            fixed_fee=0.05, # $0.05 fixed fee per execution
            base_spread=0.015, # 1.5% base spread
            latency_periods=1 # 1 step latency
        )

    def split_data(self):
        split_idx = int(len(self.data) * self.train_split)
        train_data = self.data.iloc[:split_idx]
        test_data = self.data.iloc[split_idx:]
        return train_data, test_data

    def run(self, use_test_set=True):
        train_data, test_data = self.split_data()
        target_data = test_data if use_test_set else train_data
        
        mode_str = "OUT-OF-SAMPLE (Test)" if use_test_set else "IN-SAMPLE (Train)"
        print(f"Starting REALISTIC backtest in {mode_str} mode on {len(target_data)} data points...")
        
        for idx, row in target_data.iterrows():
            # Construct a mock state (v2.0 compatible)
            # Use current row for analysis (as if we see it now)
            bid, ask = self.simulator.get_market_prices(row['market_price'])
            
            state: AgentState = {
                "current_markets": [{
                    "id": f"mkt_{idx}",
                    "question": f"Will {row['city']} temperature be above {row['threshold']}?",
                    "price_yes": ask, # Agent sees the 'ask' to buy YES
                    "price_no": 1.0 - bid, # Agent sees the 'ask' to buy NO
                    "city": row['city'],
                    "lat": row.get('lat', 40.7),
                    "lon": row.get('lon', -74.0),
                    "tz": "UTC",
                    "bid_liquidity": 500.0, # More realistic liquidity
                    "ask_liquidity": 500.0
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
                "open_positions": [],
                "in_flight_exposure": 0.0
            }

            # 1. Run Analyst
            analysis = analyst_agent(state)
            state.update(analysis)

            # 2. Run Decision
            decision = decision_agent(state)
            state.update(decision)

            # 3. Resolve Trade using Execution Simulator
            market_id = f"mkt_{idx}"
            if market_id in state["position_sizes"]:
                req_size = state["position_sizes"][market_id]
                side = state["trade_sides"][market_id]
                
                # Execute with realism
                filled_size, entry_price, fee = self.simulator.simulate_execution(req_size, side, idx, self.data)
                
                if filled_size <= 0:
                    print(f" -> Order MISSED/CANCELLED for {market_id}")
                    continue

                threshold = row['threshold']
                actual_temp = row['actual_temp']
                
                # Did we win? (Always resolves at current row's actual temp)
                condition_met = actual_temp > threshold
                won = (side == "buy_yes" and condition_met) or (side == "buy_no" and not condition_met)
                
                # PnL Calc: 
                units = filled_size / entry_price
                if won:
                    pnl = units - filled_size - fee
                else:
                    pnl = -filled_size - fee

                self.bankroll += pnl
                self.trades.append({
                    "timestamp": row['timestamp'],
                    "market": market_id,
                    "side": side,
                    "requested_size": req_size,
                    "filled_size": filled_size,
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
        print(f"REALISTIC BACKTEST RESULTS: {mode_str}")
        print("="*40)
        print(f"Final Bankroll:   ${self.bankroll:.2f}")
        print(f"Total ROI:        {roi:.2%}")
        print(f"Sharpe Ratio:     {sharpe:.2f}")
        print(f"Total Trades:     {len(df_trades)}")
        print(f"Fill Rate:        {(df_trades['filled_size'].sum() / df_trades['requested_size'].sum()):.2%}")
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
        
    # Initialize Engine
    engine = BacktestEngine(data_file, fee_pct=0.002, slippage_pct=0.005, train_split=0.7)
    
    # 1. Run In-Sample
    engine.run(use_test_set=False)
    
    # 2. Reset and Run Out-of-Sample
    engine.bankroll = DEFAULT_BANKROLL
    engine.trades = []
    engine.run(use_test_set=True)
