import sqlite3
conn = sqlite3.connect('trading_bot.db')
cursor = conn.cursor()
cursor.execute("SELECT market_id, question FROM forecast_history WHERE market_id IN ('2126654', '2126656', '2126658', '2126653') GROUP BY market_id")
print(cursor.fetchall())
conn.close()
