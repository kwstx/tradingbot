from persistence import PersistenceManager
import json
from datetime import datetime

def generate_report():
    db = PersistenceManager()
    summary = db.get_paper_summary()
    
    print("\n" + "="*40)
    print("      PAPER TRADING PERFORMANCE REPORT")
    print("="*40)
    
    if "status" in summary:
        print(f"\nStatus: {summary['status']}")
    else:
        print(f"Timestamp:       {summary['timestamp']}")
        print(f"Trades Count:    {summary['trade_count']}")
        print(f"Initial Bal:     ${summary['initial_balance']:.2f}")
        print(f"Current Bal:     ${summary['current_balance']:.2f}")
        print("-" * 40)
        
        roi_color = "\033[92m" if summary['roi_pct'] >= 0 else "\033[91m"
        reset_color = "\033[0m"
        
        print(f"ROI:             {roi_color}{summary['roi_pct']:+.2f}%{reset_color}")
        print(f"Max Drawdown:    {summary['max_drawdown_pct']:.2f}%")
        
    print("="*40 + "\n")

if __name__ == "__main__":
    generate_report()
