import pmxt
from dotenv import load_dotenv
import os

load_dotenv()

def test_market_obj():
    try:
        poly = pmxt.Polymarket()
        print("Fetching markets...")
        markets = poly.fetch_markets(query="weather")
        if markets:
            m = markets[0]
            print(f"Type: {type(m)}")
            print(f"Attributes: {dir(m)}")
            # Try to see some values
            try: print(f"Active: {m.active}")
            except: pass
            try: print(f"Tokens: {m.tokens}")
            except: pass
            try: print(f"Liquidity: {m.liquidity}")
            except: pass
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_market_obj():
