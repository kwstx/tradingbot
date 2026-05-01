import sqlite3
import os

DB_PATH = "trading_bot.db"

def get_detailed_stats():
    if not os.path.exists(DB_PATH):
        print("Database not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get total trades by mode
    cursor.execute("SELECT mode, COUNT(*) FROM trades GROUP BY mode")
    modes = cursor.fetchall()
    
    print("\n--- Trade Count by Mode ---")
    for mode, count in modes:
        print(f"{mode}: {count}")

    # Get winrate for PAPER mode
    cursor.execute("SELECT COUNT(*) FROM trades WHERE mode = 'PAPER' AND status = 'WON'")
    wins = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM trades WHERE mode = 'PAPER' AND status = 'LOST'")
    losses = cursor.fetchone()[0]
    
    total_resolved = wins + losses
    winrate = (wins / total_resolved * 100) if total_resolved > 0 else 0

    print("\n--- Paper Trading Winrate ---")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Total Resolved: {total_resolved}")
    print(f"Winrate: {winrate:.2f}%")

    # Check for pending trades
    cursor.execute("SELECT COUNT(*) FROM trades WHERE mode = 'PAPER' AND status = 'paper_executed'")
    pending = cursor.fetchone()[0]
    print(f"Pending (Executed but not resolved): {pending}")

    # Get PnL stats
    cursor.execute("SELECT SUM(pnl) FROM trades WHERE mode = 'PAPER'")
    total_pnl = cursor.fetchone()[0] or 0
    print(f"Total PnL: ${total_pnl:.2f}")

    # Average profit per winning trade
    cursor.execute("SELECT AVG(pnl) FROM trades WHERE mode = 'PAPER' AND status = 'WON'")
    avg_win = cursor.fetchone()[0] or 0
    
    # Average loss per losing trade
    cursor.execute("SELECT AVG(pnl) FROM trades WHERE mode = 'PAPER' AND status = 'LOST'")
    avg_loss = cursor.fetchone()[0] or 0

    print(f"Avg Profit (Wins): ${avg_win:.2f}")
    print(f"Avg Loss (Losses): ${avg_loss:.2f}")

    conn.close()

if __name__ == "__main__":
    get_detailed_stats()
