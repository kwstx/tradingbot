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
            pnl REAL DEFAULT 0.0,
            mode TEXT DEFAULT 'LIVE'
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bankroll_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            balance REAL,
            equity REAL,
            mode TEXT DEFAULT 'LIVE'
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bayesian_priors (
            market_id TEXT PRIMARY KEY,
            prior_prob REAL,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS api_weights (
            api_name TEXT PRIMARY KEY,
            weight REAL DEFAULT 1.0,
            last_error REAL DEFAULT 0.0,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS forecast_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            api_name TEXT,
            forecast_mu REAL,
            forecast_sigma REAL,
            threshold REAL,
            target_date TEXT,
            lat REAL,
            lon REAL,
            actual_value REAL DEFAULT NULL,
            error REAL DEFAULT NULL
        )
    ''')
    
    # Migration: Check if threshold column exists
    cursor.execute("PRAGMA table_info(forecast_history)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'threshold' not in columns:
        cursor.execute("ALTER TABLE forecast_history ADD COLUMN threshold REAL")

    # Initialize default weights for known providers
    cursor.execute("INSERT OR IGNORE INTO api_weights (api_name, weight) VALUES ('open_meteo', 1.0)")
    cursor.execute("INSERT OR IGNORE INTO api_weights (api_name, weight) VALUES ('noaa', 1.2)") # NOAA weight slightly higher in US
    
    # Initialize default paper bankroll
    cursor.execute("SELECT COUNT(*) FROM bankroll_history WHERE mode = 'PAPER'")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO bankroll_history (balance, equity, mode) VALUES (50.0, 50.0, 'PAPER')")
    
    conn.commit()
    conn.close()

class PersistenceManager:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        init_db()

    def log_trade(self, market_id, side, size, price, status, mode='LIVE'):
        """Logs a trade to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO trades (market_id, side, size_usdc, price, status, mode)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (market_id, side, size, price, status, mode))
        conn.commit()
        conn.close()

    def update_bankroll(self, balance, equity, mode='LIVE'):
        """Logs the current bankroll state."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO bankroll_history (balance, equity, mode)
            VALUES (?, ?, ?)
        ''', (balance, equity, mode))
        conn.commit()
        conn.close()

    def get_latest_bankroll(self, mode='LIVE', default=50.0):
        """Retrieves the latest bankroll for a specific mode."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT balance FROM bankroll_history 
            WHERE mode = ? 
            ORDER BY timestamp DESC LIMIT 1
        ''', (mode,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else default

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

    def get_api_weights(self):
        """Retrieves performance-based weights for all weather APIs."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT api_name, weight FROM api_weights')
        rows = cursor.fetchall()
        conn.close()
        return {row[0]: row[1] for row in rows}

    def update_api_performance(self, api_name, error_val):
        """Updates the weight based on the recent prediction error."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE api_weights 
            SET weight = CASE 
                WHEN ? < 0.5 THEN weight * 1.02 
                WHEN ? > 2.0 THEN weight * 0.98
                ELSE weight 
            END,
            last_error = ?,
            last_updated = CURRENT_TIMESTAMP
            WHERE api_name = ?
        ''', (error_val, error_val, error_val, api_name))
        conn.commit()
        conn.close()

    def save_forecast(self, market_id, api_name, mu, sigma, threshold, target_date, lat, lon):
        """Saves a forecast to the history for later backtesting."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO forecast_history (market_id, api_name, forecast_mu, forecast_sigma, threshold, target_date, lat, lon)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (market_id, api_name, mu, sigma, threshold, target_date, lat, lon))
        conn.commit()
        conn.close()

    def get_unresolved_forecasts(self):
        """Retrieves forecasts that haven't been resolved yet."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, market_id, api_name, target_date, forecast_mu, lat, lon FROM forecast_history 
            WHERE actual_value IS NULL
        ''')
        rows = cursor.fetchall()
        conn.close()
        return rows

    def resolve_forecast(self, forecast_id, actual_value):
        """Updates a forecast with its actual outcome and calculates the error."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Fetch the forecast mu to calculate error
        cursor.execute("SELECT forecast_mu, api_name FROM forecast_history WHERE id = ?", (forecast_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return
            
        mu, api_name = row
        error = abs(mu - actual_value)
        
        cursor.execute('''
            UPDATE forecast_history 
            SET actual_value = ?, error = ?
            WHERE id = ?
        ''', (actual_value, error, forecast_id))
        
        conn.commit()
        conn.close()
        
        # Also update the api weight based on this error
        self.update_api_performance(api_name, error)

    def get_unresolved_paper_trades(self):
        """Retrieves paper trades that are executed but not resolved."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Join with forecast_history to get target_date, threshold, lat, lon
        # Since we might have multiple forecasts for one market (from different APIs), 
        # we just need the ground truth from one of them.
        cursor.execute('''
            SELECT t.id, t.market_id, t.side, t.size_usdc, t.price, 
                   f.target_date, f.threshold, f.lat, f.lon 
            FROM trades t
            JOIN forecast_history f ON t.market_id = f.market_id
            WHERE t.mode = 'PAPER' AND t.status = 'paper_executed'
            GROUP BY t.id
        ''')
        rows = cursor.fetchall()
        conn.close()
        return rows

    def resolve_paper_trade(self, trade_id, size, won, pnl):
        """Updates a paper trade with win/loss result and updates bankroll."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 1. Update trade status and PnL
        status = 'WON' if won else 'LOST'
        cursor.execute('''
            UPDATE trades SET status = ?, pnl = ? WHERE id = ?
        ''', (status, pnl, trade_id))
        
        # 2. Update Bankroll
        # Fetch current latest balance
        cursor.execute("SELECT balance FROM bankroll_history WHERE mode = 'PAPER' ORDER BY timestamp DESC LIMIT 1")
        current_bal = cursor.fetchone()[0]
        
        # In paper mode, 'size' was deducted at execution.
        # If WON: We credit back the full payout (Size + Net PnL).
        # If LOST: We credit back $0 (the size is already gone).
        if won:
            new_bal = current_bal + (size + pnl)
        else:
            new_bal = current_bal
            
        cursor.execute('''
            INSERT INTO bankroll_history (balance, equity, mode)
            VALUES (?, ?, 'PAPER')
        ''', (new_bal, new_bal))
        
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

    def get_paper_summary(self):
        """Generates a performance summary for the Paper Trading session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 1. Total Paper Trades
        cursor.execute("SELECT COUNT(*) FROM trades WHERE mode = 'PAPER'")
        trade_count = cursor.fetchone()[0]
        
        # 2. Virtual Bankroll Progression
        cursor.execute("SELECT balance FROM bankroll_history WHERE mode = 'PAPER' ORDER BY timestamp ASC")
        balances = [row[0] for row in cursor.fetchall()]
        
        if not balances:
            conn.close()
            return {"status": "No paper trading data found."}
            
        initial_bal = balances[0]
        current_bal = balances[-1]
        roi = ((current_bal - initial_bal) / initial_bal) * 100 if initial_bal > 0 else 0
        
        # 3. Max Drawdown (Simple calculation)
        peak = 0
        max_dd = 0
        for b in balances:
            if b > peak: peak = b
            dd = (peak - b) / peak if peak > 0 else 0
            if dd > max_dd: max_dd = dd
            
        conn.close()
        return {
            "mode": "PAPER",
            "trade_count": trade_count,
            "initial_balance": round(initial_bal, 2),
            "current_balance": round(current_bal, 2),
            "roi_pct": round(roi, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "timestamp": datetime.now().isoformat()
        }
