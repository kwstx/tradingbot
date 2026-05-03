import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
from backtest import BacktestEngine, ExecutionSimulator
from weather_arb_agent import DEFAULT_BANKROLL, AgentState, analyst_agent, decision_agent

def run_permutation_test(data_file="actual_weather_data.json", n_permutations=100):
    """
    Step 5: Randomization / Permutation Test
    Tests if the strategy's alpha is genuine or just noise by shuffling 
    signals and re-calculating performance.
    """
    if not os.path.exists(data_file):
        print(f"Data file {data_file} not found. Please run backtest.py first to generate data.")
        return

    # 1. Initialize Engine and Data
    engine = BacktestEngine(data_file)
    train_data, test_data = engine.split_data()
    # We use the Out-of-Sample test data for the pro-level check
    data = test_data.copy().reset_index(drop=True)
    
    print(f"\n[STEP 1] Generating REAL signals for {len(data)} OOS data points...")
    real_decisions = []
    
    # 2. Extract Real Signals
    # We run the agents on the historical data to see what they WOULD have done
    for idx, row in data.iterrows():
        bid, ask = engine.simulator.get_market_prices(row['market_price'])
        
        # Construct mock state for the agent
        state: AgentState = {
            "current_markets": [{
                "id": f"mkt_{idx}",
                "question": f"Will {row['city']} temperature be above {row['threshold']}?",
                "price_yes": ask,
                "price_no": 1.0 - bid,
                "city": row['city'],
                "lat": row.get('lat', 40.7),
                "lon": row.get('lon', -74.0),
                "tz": "UTC",
                "bid_liquidity": 500.0,
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
            "bankroll": DEFAULT_BANKROLL,
            "api_weights": {"open_meteo": 1.0},
            "open_positions": [],
            "in_flight_exposure": 0.0
        }
        
        # Run the same logic as the live bot
        analysis = analyst_agent(state)
        state.update(analysis)
        decision = decision_agent(state)
        
        market_id = f"mkt_{idx}"
        if market_id in decision["position_sizes"]:
            real_decisions.append({
                "side": decision["trade_sides"][market_id],
                "size": decision["position_sizes"][market_id]
            })
        else:
            real_decisions.append(None)

    # 3. Performance Calculation Function
    def calculate_performance(decisions, data_df):
        """Calculates PnL for a sequence of decisions against a dataset."""
        total_pnl = 0
        trades_count = 0
        
        for idx, decision in enumerate(decisions):
            if decision is None:
                continue
            
            row = data_df.iloc[idx]
            side = decision["side"]
            size = decision["size"]
            
            # Use deterministic execution for fair comparison across permutations
            bid, ask = engine.simulator.get_market_prices(row['market_price'])
            price = ask if side == "buy_yes" else (1.0 - bid)
            
            # Simple PnL: (Outcome / Price) - 1
            condition_met = row['actual_temp'] > row['threshold']
            won = (side == "buy_yes" and condition_met) or (side == "buy_no" and not condition_met)
            
            fee = size * engine.simulator.fee_pct
            units = size / price
            pnl = (units - size - fee) if won else (-size - fee)
            
            total_pnl += pnl
            trades_count += 1
            
        return total_pnl, trades_count

    # 4. Compute Real Performance
    real_pnl, real_trades = calculate_performance(real_decisions, data)
    print(f" -> Real Performance: ${real_pnl:.2f} over {real_trades} trades.")

    # 5. Run Permutations (The "Fake Alpha" Test)
    print(f"\n[STEP 2] Running {n_permutations} permutations (shuffling signals)...")
    shuffled_results = []
    
    for i in range(n_permutations):
        # Shuffle the signal sequence
        shuffled_decisions = real_decisions.copy()
        np.random.shuffle(shuffled_decisions)
        
        pnl, _ = calculate_performance(shuffled_decisions, data)
        shuffled_results.append(pnl)
        
        if (i + 1) % (n_permutations // 10 or 1) == 0:
            print(f" -> Progress: {i+1}/{n_permutations} permutations complete.")

    shuffled_results = np.array(shuffled_results)
    
    # 6. Statistical Analysis
    p_value = np.mean(shuffled_results >= real_pnl)
    z_score = (real_pnl - shuffled_results.mean()) / shuffled_results.std() if shuffled_results.std() > 0 else 0
    
    print("\n" + "="*50)
    print("PRO-LEVEL CHECK: RANDOMIZATION / PERMUTATION TEST")
    print("="*50)
    print(f"REAL STRATEGY PNL:    ${real_pnl:.2f}")
    print(f"SHUFFLED MEAN PNL:    ${shuffled_results.mean():.2f}")
    print(f"SHUFFLED MAX PNL:     ${shuffled_results.max():.2f}")
    print(f"SHUFFLED STD DEV:     ${shuffled_results.std():.2f}")
    print("-" * 50)
    print(f"Z-SCORE:              {z_score:.2f}")
    print(f"P-VALUE:              {p_value:.4f}")
    print("-" * 50)
    
    if p_value < 0.01:
        print("RESULT: ALPHA IS ELITE (Legendary conviction)")
    elif p_value < 0.05:
        print("RESULT: ALPHA IS GENUINE (Significant edge detected)")
    else:
        print("RESULT: ALPHA IS NOISE (Fake Alpha detected!)")
    print("="*50)

    # 7. Visualization
    plt.figure(figsize=(12, 7))
    plt.style.use('dark_background')
    
    # Histogram of shuffled results
    n, bins, patches = plt.hist(shuffled_results, bins=25, alpha=0.6, color='#00d4ff', label='Shuffled (Noise) Distribution')
    
    # Mark the real PnL
    plt.axvline(real_pnl, color='#ff007b', linestyle='--', linewidth=3, label=f'Real Strategy (${real_pnl:.2f})')
    
    # Aesthetics
    plt.title('Permutation Test: Real Alpha vs. Random Noise', fontsize=16, fontweight='bold', color='white', pad=20)
    plt.xlabel('Total PnL ($)', fontsize=12, color='#cccccc')
    plt.ylabel('Frequency', fontsize=12, color='#cccccc')
    plt.grid(axis='y', alpha=0.3)
    plt.legend(facecolor='#222222', edgecolor='#444444')
    
    # Annotations
    plt.annotate(f'P-Value: {p_value:.4f}', xy=(real_pnl, plt.ylim()[1]*0.8), xytext=(real_pnl*1.1 if real_pnl > 0 else real_pnl*0.9, plt.ylim()[1]*0.9),
                 arrowprops=dict(facecolor='white', shrink=0.05, width=1, headwidth=8),
                 fontsize=12, fontweight='bold', color='#ff007b')

    plt.tight_layout()
    plt.savefig('permutation_test_report.png', dpi=150)
    print(f"\nDetailed report plot saved as 'permutation_test_report.png'")

if __name__ == "__main__":
    # Ensure data exists by running a quick dummy check if file is missing
    # (Though we expect the user to have run backtest.py)
    run_permutation_test(n_permutations=1000)
