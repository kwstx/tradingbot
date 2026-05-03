import sqlite3
conn = sqlite3.connect('trading_bot.db')
cursor = conn.cursor()
cursor.execute('''
    SELECT t.market_id, t.side, t.price, f.threshold 
    FROM trades t 
    JOIN forecast_history f ON t.market_id = f.market_id 
    WHERE t.mode = 'PAPER' 
    GROUP BY t.id
''')
print(cursor.fetchall())
conn.close()
