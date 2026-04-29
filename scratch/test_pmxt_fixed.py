import pmxt
from dotenv import load_dotenv
import os

load_dotenv()

def test_pmxt_correct():
    try:
        # Based on dir(pmxt), let's try Polymarket (capital P)
        print("Testing pmxt.Polymarket()...")
        poly = pmxt.Polymarket()
        print("Successfully initialized Polymarket class.")
        
        # Try to fetch markets
        print("Fetching markets...")
        markets = poly.fetch_markets(query="weather")
        print(f"Found {len(markets)} weather markets.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_pmxt_correct()
