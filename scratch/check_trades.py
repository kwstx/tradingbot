import sqlite3
import pandas as pd

def check_trades():
    conn = sqlite3.connect('trading_bot.db')
    print("--- Last 20 Trades ---")
    trades = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 20", conn)
    print(trades)
    conn.close()

if __name__ == "__main__":
    check_trades()
