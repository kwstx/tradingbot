import sqlite3
conn = sqlite3.connect('trading_bot.db')
cursor = conn.cursor()
cursor.execute("SELECT status, COUNT(*) FROM trades WHERE mode = 'PAPER' GROUP BY status")
print(cursor.fetchall())
conn.close()
