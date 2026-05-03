import sqlite3

db_path = "trading_bot.db"

def reset_paper_trading():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("Clearing Paper Trading data...")
    
    # 1. Clear trades
    cursor.execute("DELETE FROM trades WHERE mode = 'PAPER'")
    print(f" -> Removed {cursor.rowcount} paper trades.")
    
    # 2. Clear bankroll history
    cursor.execute("DELETE FROM bankroll_history WHERE mode = 'PAPER'")
    print(f" -> Removed {cursor.rowcount} bankroll records.")
    
    # 3. Reset to initial $50.00
    cursor.execute("INSERT INTO bankroll_history (balance, equity, mode) VALUES (50.0, 50.0, 'PAPER')")
    print(" -> Initialized Paper Bankroll to $50.00.")
    
    # 4. Clear forecasts (optional but cleaner)
    # cursor.execute("DELETE FROM forecast_history")
    
    conn.commit()
    conn.close()
    print("Reset complete.")

if __name__ == "__main__":
    reset_paper_trading()
