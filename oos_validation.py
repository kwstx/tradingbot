import os
import json
import pandas as pd
import numpy as np
import hashlib
import argparse
from datetime import datetime
from backtest import BacktestEngine, DEFAULT_BANKROLL

# --- CONFIGURATION ---
DATA_FILE = "actual_weather_data.json"
TRAIN_TEST_SPLIT_DATE = "2026-04-18" # Cutoff for OOS testing
LOCK_FILE = ".oos_lock"

def get_data_hash(df):
    """Generates a hash of the dataframe to ensure data integrity."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

def run_validation(mode):
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    # Load data
    df = pd.read_json(DATA_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    # Split data
    train_df = df[df['timestamp'] <= TRAIN_TEST_SPLIT_DATE]
    test_df = df[df['timestamp'] > TRAIN_TEST_SPLIT_DATE]

    if mode == "train":
        print(f"\n[PHASE 1] IN-SAMPLE TRAINING (Optimization)")
        print(f"Period: {train_df['timestamp'].min().date()} to {train_df['timestamp'].max().date()}")
        print(f"Data Points: {len(train_df)}")
        
        engine = BacktestEngine(DATA_FILE)
        engine.data = train_df # Override with train set
        engine.run(use_test_set=False) # The label in engine.run will say In-Sample
        
    elif mode == "test":
        print(f"\n[PHASE 2] OUT-OF-SAMPLE TESTING (The Moment of Truth)")
        print(f"Period: {test_df['timestamp'].min().date()} to {test_df['timestamp'].max().date()}")
        print(f"Data Points: {len(test_df)}")
        
        # Security Check: Has the user peeked?
        if os.path.exists(LOCK_FILE):
            with open(LOCK_FILE, "r") as f:
                lock_data = json.load(f)
                if lock_data.get("peeked_test"):
                    print("⚠️ WARNING: Test data has already been viewed. Results may be biased.")
        
        engine = BacktestEngine(DATA_FILE)
        engine.data = test_df # Override with test set
        engine.run(use_test_set=True)
        
        # Mark as peeked
        with open(LOCK_FILE, "w") as f:
            json.dump({"peeked_test": True, "last_test_date": datetime.now().isoformat()}, f)

    else:
        print("Invalid mode. Use 'train' or 'test'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Out-of-Sample Validation Tool")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"], help="Validation mode")
    args = parser.parse_args()
    
    run_validation(args.mode)
