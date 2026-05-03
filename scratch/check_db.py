import sqlite3
import pandas as pd
from datetime import datetime

db_path = "trading_bot.db"

def check_status():
    conn = sqlite3.connect(db_path)
    
    print("--- Bankroll History (Last 5) ---")
    df_bankroll = pd.read_sql_query("SELECT * FROM bankroll_history WHERE mode = 'PAPER' ORDER BY timestamp DESC LIMIT 5", conn)
    print(df_bankroll)
    
    print("\n--- Recent Trades (Last 10) ---")
    df_trades = pd.read_sql_query("SELECT * FROM trades WHERE mode = 'PAPER' ORDER BY timestamp DESC LIMIT 10", conn)
    print(df_trades)
    
    print("\n--- Trade Status Summary ---")
    df_status = pd.read_sql_query("SELECT status, count(*), sum(pnl) FROM trades WHERE mode = 'PAPER' GROUP BY status", conn)
    print(df_status)

    print("\n--- API Weight Performance ---")
    df_weights = pd.read_sql_query("SELECT * FROM api_weights", conn)
    print(df_weights)

    conn.close()

if __name__ == "__main__":
    check_status()
