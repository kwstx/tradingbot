import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from backtest import BacktestEngine, DEFAULT_BANKROLL

class BaselineComparison:
    def __init__(self, data_path, fee_pct=0.002, slippage_pct=0.005):
        self.data_path = data_path
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.data = pd.read_json(data_path)
        
    def run_system_backtest(self):
        """Runs the actual trading bot logic."""
        engine = BacktestEngine(self.data_path, fee_pct=self.fee_pct, slippage_pct=self.slippage_pct, train_split=0.0)
        # We use train_split=0 to run on the entire dataset for comparison
        engine.run(use_test_set=True) 
        
        df_trades = pd.DataFrame(engine.trades)
        if df_trades.empty:
            return {"name": "Current System", "roi": 0, "win_rate": 0, "trades": 0, "sharpe": 0}
            
        roi = (engine.bankroll - DEFAULT_BANKROLL) / DEFAULT_BANKROLL
        returns = df_trades['pnl'] / DEFAULT_BANKROLL
        sharpe = (returns.mean() / returns.std()) * np.sqrt(365) if len(returns) > 1 and returns.std() > 0 else 0
        
        return {
            "name": "Current System (Agentic)",
            "roi": roi,
            "win_rate": (df_trades['won'].sum() / len(df_trades)),
            "trades": len(df_trades),
            "sharpe": sharpe,
            "final_bankroll": engine.bankroll
        }

    def run_baseline(self, strategy_type, threshold_x=0.15):
        """Runs a 'dumb' baseline strategy."""
        bankroll = DEFAULT_BANKROLL
        trades = []
        
        for idx, row in self.data.iterrows():
            price_yes = row['market_price']
            price_no = 1.0 - price_yes
            threshold = row['threshold']
            mu = row['forecast_mu']
            sigma = row['forecast_sigma']
            
            # Calculate naive ensemble probability: P(actual_temp > threshold)
            ensemble_prob = 1.0 - norm.cdf(threshold, loc=mu, scale=sigma)
            
            side = None
            if strategy_type == "market_momentum":
                # Always trade market price probability (directional consensus)
                side = "buy_yes" if price_yes > 0.5 else "buy_no"
            
            elif strategy_type == "extreme_mispricing":
                # Always take "extreme mispricing > X%"
                if abs(ensemble_prob - price_yes) > threshold_x:
                    side = "buy_yes" if ensemble_prob > price_yes else "buy_no"
            
            elif strategy_type == "ensemble_mean":
                # Always follow ensemble mean forecast
                side = "buy_yes" if ensemble_prob > 0.5 else "buy_no"
            
            if side:
                # Use a fixed size for baselines (15% of bankroll, matching new MAX_POSITION_SIZE_PCT)
                size = bankroll * 0.15
                
                # Apply the same constraints as the system for fairness
                # 1. Max Position Size
                # 2. Liquidity Cap (25% of 1000 = 250)
                liquidity_limit = 250.0 
                size = min(size, liquidity_limit)
                
                # Slippage and Fee
                if side == "buy_yes":
                    entry_price = price_yes + (price_yes * self.slippage_pct)
                else:
                    entry_price = price_no + (price_no * self.slippage_pct)
                
                entry_price = min(0.999, entry_price)
                fee = size * self.fee_pct
                
                # Resolution
                actual_temp = row['actual_temp']
                condition_met = actual_temp > threshold
                won = (side == "buy_yes" and condition_met) or (side == "buy_no" and not condition_met)
                
                units = size / entry_price
                if won:
                    pnl = units - size - fee
                else:
                    pnl = -size - fee
                
                bankroll += pnl
                trades.append({
                    "won": won,
                    "pnl": pnl,
                    "bankroll": bankroll
                })
        
        df_trades = pd.DataFrame(trades)
        if df_trades.empty:
            return {"name": strategy_type, "roi": 0, "win_rate": 0, "trades": 0, "sharpe": 0}
            
        roi = (bankroll - DEFAULT_BANKROLL) / DEFAULT_BANKROLL
        returns = df_trades['pnl'] / DEFAULT_BANKROLL
        sharpe = (returns.mean() / returns.std()) * np.sqrt(365) if len(returns) > 1 and returns.std() > 0 else 0
        
        return {
            "name": f"Baseline: {strategy_type}",
            "roi": roi,
            "win_rate": (df_trades['won'].sum() / len(df_trades)),
            "trades": len(df_trades),
            "sharpe": sharpe,
            "final_bankroll": bankroll
        }

    def generate_report(self):
        results = []
        results.append(self.run_system_backtest())
        results.append(self.run_baseline("market_momentum"))
        results.append(self.run_baseline("extreme_mispricing", threshold_x=0.15))
        results.append(self.run_baseline("ensemble_mean"))
        
        df_report = pd.DataFrame(results)
        
        print("\n" + "="*80)
        print("STRATEGY ALPHA COMPARISON REPORT")
        print("="*80)
        print(df_report[['name', 'roi', 'win_rate', 'trades', 'sharpe']].to_string(index=False, formatters={
            'roi': '{:,.2%}'.format,
            'win_rate': '{:,.2%}'.format,
            'sharpe': '{:,.2f}'.format
        }))
        print("="*80)
        
        # Check for Alpha
        system_roi = results[0]['roi']
        best_baseline_roi = max(r['roi'] for r in results[1:])
        
        if system_roi > best_baseline_roi:
            print(f"VERDICT: POSITIVE ALPHA (+{(system_roi - best_baseline_roi):.2%} over best baseline)")
        else:
            print(f"VERDICT: NO ALPHA DETECTED (System underperforms best baseline by {(best_baseline_roi - system_roi):.2%})")
        print("="*80 + "\n")

if __name__ == "__main__":
    data_file = "actual_weather_data.json"
    comparator = BaselineComparison(data_file)
    comparator.generate_report()
