import sqlite3
import os

DB_PATH = "trading_bot.db"

def reset_database():
    if not os.path.exists(DB_PATH):
        print("Database not found. Nothing to reset.")
        return

    print(f"Resetting database: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Clear performance tables
        cursor.execute("DELETE FROM trades")
        cursor.execute("DELETE FROM bankroll_history")
        cursor.execute("DELETE FROM forecast_history")
        cursor.execute("DELETE FROM bayesian_priors")
        
        # Keep api_weights but reset them to default 1.0
        cursor.execute("UPDATE api_weights SET weight = 1.0, last_error = 0.0")

        # Initialize fresh paper bankroll
        cursor.execute("INSERT INTO bankroll_history (balance, equity, mode) VALUES (50.0, 50.0, 'PAPER')")
        
        conn.commit()
        print("DONE: Database successfully reset.")
        print("DONE: Initial Paper Balance set to $50.00.")
        print("DONE: History cleared. Hardened session ready.")
        
    except Exception as e:
        print(f"ERROR resetting database: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    reset_database()
