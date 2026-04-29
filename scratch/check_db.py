import sqlite3
import pandas as pd
from datetime import datetime, timedelta

def check_bot_status():
    conn = sqlite3.connect('trading_bot.db')
    
    print("--- Last 10 Trades ---")
    trades = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 10", conn)
    print(trades)
    
    print("\n--- Last 10 Bankroll Updates ---")
    bankroll = pd.read_sql_query("SELECT * FROM bankroll_history ORDER BY timestamp DESC LIMIT 10", conn)
    print(bankroll)
    
    print("\n--- Recent Logs/Errors (if any) ---")
    # Checking if there's a logs table
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"Tables: {tables}")
    
    if ('logs',) in tables:
        logs = pd.read_sql_query("SELECT * FROM logs ORDER BY timestamp DESC LIMIT 10", conn)
        print(logs)
    
    if ('priors',) in tables:
        print("\n--- Last 5 Priors ---")
        priors = pd.read_sql_query("SELECT * FROM priors ORDER BY timestamp DESC LIMIT 5", conn)
        print(priors)

    if ('forecast_history',) in tables:
        print("\n--- Last 5 Forecasts ---")
        forecasts = pd.read_sql_query("SELECT * FROM forecast_history ORDER BY timestamp DESC LIMIT 5", conn)
        print(forecasts)

    conn.close()

if __name__ == "__main__":
    check_bot_status()
