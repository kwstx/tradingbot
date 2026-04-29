import pmxt
from dotenv import load_dotenv
import os

load_dotenv()

def test_pmxt():
    try:
        poly = pmxt.polymarket()
        print("Fetching markets...")
        markets = poly.fetch_markets(query="weather")
        print(f"Found {len(markets)} weather markets.")
        for m in markets[:5]:
            print(f"- {m.get('question')} (Active: {m.get('active')}, Liquidity: {m.get('liquidity')})")
            
        print("\nChecking balance...")
        try:
            balance = poly.fetch_balance()
            print(f"Balance: {balance}")
        except Exception as e:
            print(f"Balance fetch failed: {e}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_pmxt()
