import sqlite3
import json
from datetime import datetime
import os

DB_PATH = "trading_bot.db"

def init_db():
    """Initializes the SQLite database with required tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Tables for trades, bankroll, and priors
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            market_id TEXT,
            side TEXT,
            size_usdc REAL,
            price REAL,
            status TEXT,
            pnl REAL DEFAULT 0.0
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bankroll_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            balance REAL,
            equity REAL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bayesian_priors (
            market_id TEXT PRIMARY KEY,
            prior_prob REAL,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

class PersistenceManager:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        if not os.path.exists(self.db_path):
            init_db()

    def log_trade(self, market_id, side, size, price, status):
        """Logs a trade to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO trades (market_id, side, size_usdc, price, status)
            VALUES (?, ?, ?, ?, ?)
        ''', (market_id, side, size, price, status))
        conn.commit()
        conn.close()

    def update_bankroll(self, balance, equity):
        """Logs the current bankroll state."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO bankroll_history (balance, equity)
            VALUES (?, ?)
        ''', (balance, equity))
        conn.commit()
        conn.close()

    def get_prior(self, market_id, default=0.5):
        """Retrieves the Bayesian prior for a specific market."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT prior_prob FROM bayesian_priors WHERE market_id = ?', (market_id,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else default

    def save_prior(self, market_id, prob):
        """Saves/Updates the Bayesian prior for a specific market."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO bayesian_priors (market_id, prior_prob, last_updated)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(market_id) DO UPDATE SET 
                prior_prob = excluded.prior_prob,
                last_updated = CURRENT_TIMESTAMP
        ''', (market_id, prob))
        conn.commit()
        conn.close()

    def get_daily_summary(self):
        """Generates a summary for the last 24 hours."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Simple stats
        cursor.execute("SELECT COUNT(*) FROM trades WHERE timestamp > datetime('now', '-1 day')")
        trade_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT balance FROM bankroll_history ORDER BY timestamp DESC LIMIT 1")
        last_balance = cursor.fetchone()
        balance = last_balance[0] if last_balance else 0.0
        
        conn.close()
        return {
            "trade_count": trade_count,
            "current_balance": balance,
            "timestamp": datetime.now().isoformat()
        }
