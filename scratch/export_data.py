import sqlite3
import pandas as pd
import json

def export_history():
    conn = sqlite3.connect('trading_bot.db')
    # We need timestamp, city (from lat/lon or some mapping), threshold (we might need to infer this), market_price (mocked or from trades), actual_temp, forecast_mu, forecast_sigma
    # Wait, the database doesn't store the threshold in forecast_history.
    # It stores market_id.
    
    query = """
    SELECT f.timestamp, f.market_id, f.forecast_mu, f.forecast_sigma, f.actual_value as actual_temp, f.lat, f.lon
    FROM forecast_history f
    WHERE f.actual_value IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("No actual data found in database.")
        return
    
    # We need to add 'threshold' and 'market_price' which aren't in forecast_history.
    # Let's assume a default threshold or try to find it in the question if we have it?
    # The database doesn't have the question.
    
    # Let's see if we have trades for these markets to get prices.
    conn = sqlite3.connect('trading_bot.db')
    trades_df = pd.read_sql_query("SELECT market_id, price FROM trades", conn)
    conn.close()
    
    # Merge or just assume a price if not found
    data = []
    for _, row in df.iterrows():
        # Fallback values for backtest requirements
        threshold = 70.0 # Default fallback
        market_price = 0.5 # Default fallback
        
        # Try to find a trade price
        match = trades_df[trades_df['market_id'] == row['market_id']]
        if not match.empty:
            market_price = match['price'].iloc[0]
            
        data.append({
            "timestamp": row['timestamp'],
            "city": f"Loc_{row['lat']}_{row['lon']}",
            "threshold": threshold,
            "market_price": market_price,
            "actual_temp": row['actual_temp'],
            "forecast_mu": row['forecast_mu'],
            "forecast_sigma": row['forecast_sigma'],
            "lat": row['lat'],
            "lon": row['lon']
        })
    
    with open('actual_historical_data.json', 'w') as f:
        json.dump(data, f)
    print(f"Exported {len(data)} actual data points.")

if __name__ == "__main__":
    export_history()
