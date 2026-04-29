import sqlite3
import pandas as pd

def check_priors():
    conn = sqlite3.connect('trading_bot.db')
    print("--- Bayesian Priors ---")
    priors = pd.read_sql_query("SELECT * FROM bayesian_priors ORDER BY last_updated DESC LIMIT 20", conn)
    print(priors)
    conn.close()

if __name__ == "__main__":
    check_priors()
