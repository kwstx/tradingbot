import sqlite3
from persistence import PersistenceManager

conn = sqlite3.connect('trading_bot.db')
cursor = conn.cursor()
cursor.execute('''
    SELECT t.id, t.market_id, t.status, f.target_date, t.size_usdc
    FROM trades t
    JOIN forecast_history f ON t.market_id = f.market_id
    WHERE t.mode = 'PAPER' AND t.status = 'paper_executed'
    GROUP BY t.id
''')
trades = cursor.fetchall()
print("Unresolved Paper Trades with Target Dates:")
for t in trades:
    print(t)
conn.close()
