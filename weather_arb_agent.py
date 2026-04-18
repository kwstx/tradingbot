import os
import time
import json
import re
import numpy as np
import scipy.stats as stats
import schedule
from typing import TypedDict, List, Dict, Any, Union
from pydantic import BaseModel, Field
from dotenv import load_dotenv

import httpx
import concurrent.futures
from datetime import datetime

# PMXT and LangGraph imports
try:
    import pmxt
    from langgraph.graph import StateGraph, END
except ImportError:
    # Stubs for environment resilience
    class StateGraph:
        def __init__(self, state_schema): self.nodes = {}
        def add_node(self, name, func): self.nodes[name] = func
        def set_entry_point(self, name): pass
        def add_edge(self, src, dst): pass
        def add_conditional_edges(self, src, func, mapping): pass
        def compile(self): return self
        def invoke(self, state): return state
    END = "__end__"

from persistence import PersistenceManager

load_dotenv()

# --- Simulation Mode Flag ---
SIMULATION_MODE = os.getenv("SIMULATION_MODE", "true").lower() == "true"
HUMAN_APPROVAL_REQUIRED = os.getenv("HUMAN_APPROVAL_REQUIRED", "true").lower() == "true"

# Initialize Persistence
db = PersistenceManager()


# --- 1. Shared State Definition ---
class AgentState(TypedDict):
    """
    Shared state dictionary passed between agents.
    Contains fields for market data, quantitative analysis, and risk management.
    """
    current_markets: List[Dict[str, Any]]
    weather_forecasts: Dict[str, Any]
    probabilities: Dict[str, float]
    edge_deltas: Dict[str, float]
    ev_values: Dict[str, float]
    position_sizes: Dict[str, float]
    risk_flags: List[str]
    human_approval: bool
    cycle_logs: List[str]
    pause_flag: bool
    total_exposure: float
    trade_sides: Dict[str, str]

# --- 2. Notifications ---
def send_telegram_msg(message: str):
    """Sends a notification to the user via Telegram."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print(f"[NOTIFY] Telegram credentials missing, would send: {message}")
        return
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        with httpx.Client() as client:
            client.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"})
    except Exception as e:
        print(f"[ERROR] Failed to send Telegram message: {e}")

# --- 3. Specialized Agent Nodes ---

# --- City/Coordinates Mapping ---
CITY_COORDS = {
    "NYC": {"lat": 40.7128, "lon": -74.0060, "tz": "America/New_York"},
    "London": {"lat": 51.5074, "lon": -0.1278, "tz": "Europe/London"},
    "Tokyo": {"lat": 35.6895, "lon": 139.6917, "tz": "Asia/Tokyo"},
    "Chicago": {"lat": 41.8781, "lon": -87.6298, "tz": "America/Chicago"},
}

def fetch_weather_data(lat: float, lon: float, tz: str) -> Dict[str, Any]:
    """
    Fetches ensemble forecast from Open-Meteo and supplements with NOAA if in US.
    """
    results = {}
    try:
        # Open-Meteo call
        om_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m&timezone={tz}"
        with httpx.Client() as client:
            resp = client.get(om_url, timeout=10.0)
            if resp.status_code == 200:
                results["open_meteo"] = resp.json()
                
        # Supplement with NOAA if coordinates are in US (rough check)
        if 24 < lat < 50 and -125 < lon < -66:
            noaa_points_url = f"https://api.weather.gov/points/{lat},{lon}"
            headers = {"User-Agent": "WeatherArbBot/1.0 (contact@example.com)"}
            with httpx.Client(headers=headers) as client:
                p_resp = client.get(noaa_points_url, timeout=10.0)
                if p_resp.status_code == 200:
                    forecast_url = p_resp.json()["properties"]["forecast"]
                    f_resp = client.get(forecast_url, timeout=10.0)
                    if f_resp.status_code == 200:
                        results["noaa"] = f_resp.json()
    except Exception as e:
        print(f"[ERROR] Weather API call failed: {e}")
    
    return results

def researcher_agent(state: AgentState) -> Dict[str, Any]:
    """
    Researcher Node:
    Fetches active weather markets from Polymarket using the pmxt SDK
    and correlates them with real-time weather forecasts fetched in parallel.
    """
    print("\n[RESEARCHER] Fetching active weather markets and forecasts...")
    
    try:
        # Initialize Polymarket client via pmxt
        # Note: poly.fetch_markets returns all markets, we query/filter for weather
        poly = pmxt.polymarket() 
        all_markets = poly.fetch_markets(query="weather")
        
        # Filter for active weather markets and extract details
        weather_markets = []
        unique_locations = set()
        
        for m in all_markets:
            # Basic validation: check for tokens, price, and liquidity
            if m.get('active') and m.get('tokens') and m.get('liquidity'):
                # Heuristic extraction of City and metadata
                question = m.get('question', '').upper()
                market_details = {
                    "id": m.get('id'),
                    "question": m.get('question'),
                    "current_odds": m.get('tokens', [{}])[0].get('price', 0.5), 
                    "liquidity": m.get('liquidity', 0),
                    "yes_token_id": m.get('tokens', [{}, {}])[0].get('token_id'),
                    "no_token_id": m.get('tokens', [{}, {}])[1].get('token_id')
                }
                
                # Determine city for weather lookup
                city_key = "NYC" # Default
                for city in CITY_COORDS.keys():
                    if city.upper() in question:
                        city_key = city
                        break
                
                market_details["city"] = city_key
                weather_markets.append(market_details)
                unique_locations.add(city_key)

        print(f" -> Found {len(weather_markets)} relevant weather markets.")

        # Fetch weather data in parallel for each unique location
        weather_data_map = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_city = {
                executor.submit(fetch_weather_data, CITY_COORDS[c]["lat"], CITY_COORDS[c]["lon"], CITY_COORDS[c]["tz"]): c 
                for c in unique_locations if c in CITY_COORDS
            }
            for future in concurrent.futures.as_completed(future_to_city):
                city = future_to_city[future]
                try:
                    weather_data_map[city] = future.result()
                except Exception as exc:
                    print(f" -> {city} forecast generated an exception: {exc}")

        return {
            "current_markets": weather_markets,
            "weather_forecasts": weather_data_map,
            "cycle_logs": state.get("cycle_logs", []) + [f"Researcher: {len(weather_markets)} markets and {len(weather_data_map)} forecasts ingested."]
        }
    except Exception as e:
        print(f"[CRITICAL ERROR] Researcher node failed: {e}")
        return {"cycle_logs": state.get("cycle_logs", []) + [f"Researcher error: {str(e)}"]}

def load_priors() -> Dict[str, float]:
    """Deprecated: using PersistenceManager."""
    return {}

def save_priors(priors: Dict[str, float]):
    """Deprecated: using PersistenceManager."""
    pass

def analyst_agent(state: AgentState) -> Dict[str, Any]:
    """
    Analyst Node:
    Processes weather forecasts as Gaussian ensembles and applies Bayesian updating.
    Calculates the 'fair' probability and edge delta vs Polymarket pricing.
    """
    print("[ANALYST] Running quantitative ensemble models and Bayesian updates...")
    
    p_models = {}
    edge_deltas = {}
    
    markets = state.get("current_markets", [])
    forecasts = state.get("weather_forecasts", {})
    
    for market in markets:
        m_id = market["id"]
        question = market["question"]
        city = market.get("city")
        
        # 1. Extract Thresholds (Bounds) from Question
        # Supports "above X", "below X", and "X-Y" buckets
        upper_bound = float('inf')
        lower_bound = float('-inf')
        
        # Check for range/bucket: "70-75", "70 to 75", "between 70 and 75"
        bucket_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:-|to|and)\s*(\d+(?:\.\d+)?)', question)
        if bucket_match:
            lower_bound = float(bucket_match.group(1))
            upper_bound = float(bucket_match.group(2))
        else:
            # Check for direction
            match_above = re.search(r'(?:above|over|>|greater than)\s+(\d+(?:\.\d+)?)', question, re.I)
            match_below = re.search(r'(?:below|under|<|less than)\s+(\d+(?:\.\d+)?)', question, re.I)
            if match_above:
                lower_bound = float(match_above.group(1))
            elif match_below:
                upper_bound = float(match_below.group(1))
            else:
                # Fallback: find any degree mention
                match_deg = re.search(r'(\d+(?:\.\d+)?)\b', question)
                if match_deg:
                    lower_bound = float(match_deg.group(1))

        # 2. Extract Ensemble Samples for the City
        city_data = forecasts.get(city, {})
        hourly_temps = city_data.get("open_meteo", {}).get("hourly", {}).get("temperature_2m", [])
        
        if not hourly_temps:
            print(f" -> Skipping {m_id}: No ensemble data for {city}")
            continue

        # Treat hourly forecasts as an ensemble over the next 24 hours (default window)
        samples = np.array(hourly_temps[:24])
        mu = np.mean(samples)
        sigma = np.std(samples)
        if sigma < 0.1: sigma = 0.1 # Stability constant

        # 3. Calculate Model Probability (Likelihood) using CDF
        # Calculation: P(lower < X < upper) = CDF(upper) - CDF(lower)
        # Using standardized inputs as requested
        z_upper = (upper_bound - mu) / sigma
        z_lower = (lower_bound - mu) / sigma
        
        likelihood = stats.norm.cdf(z_upper) - stats.norm.cdf(z_lower)
        likelihood = max(0.0001, min(0.9999, likelihood)) # Bound for Bayes

        # 4. Bayesian Updating
        # posterior = (likelihood * prior) / evidence
        prior = db.get_prior(m_id) # Default prior of 0.5 from DB
        evidence = (likelihood * prior) + ((1 - likelihood) * (1 - prior))
        
        p_model = (likelihood * prior) / evidence if evidence > 0 else likelihood
        
        # Persist posterior as the new prior for the next cycle
        db.save_prior(m_id, float(p_model))
        
        # 5. Edge Detection
        # Polymarket P_implied = 1 - price (as per user instruction)
        price_yes = market.get("current_odds", 0.5)
        p_market = 1.0 - price_yes
        
        delta = p_model - p_market
        
        p_models[m_id] = round(float(p_model), 4)
        edge_deltas[m_id] = round(float(delta), 4)
        
        print(f" -> Market: {m_id} | Post: {p_model:.2f} | Mkt: {p_market:.2f} | Edge: {delta:+.2f}")

    # db.save_prior handled per market during loop
    
    return {
        "probabilities": p_models,
        "edge_deltas": edge_deltas,
        "cycle_logs": state.get("cycle_logs", []) + [f"Analyst: Refined {len(p_models)} probabilities vs market pricing."]
    }

def decision_agent(state: AgentState) -> Dict[str, Any]:
    """
    Decision Node:
    Applies Expected Value (EV) and Fractional Kelly sizing to filter trades.
    Formula: EV = (p_model * (1 - market_price)) - ((1 - p_model) * market_price)
    Sizing: f_star = (b * p - q) / b with conservative k factor of 0.125.
    Cap: 15% of balance or 8 USDC max.
    """
    print("[DECISION] Applying EV filtering and fractional Kelly sizing...")
    
    probs = state["probabilities"]
    markets = state.get("current_markets", [])
    ev_values = {}
    position_sizes = {}
    trade_sides = {} # Added to track if we buy YES or NO
    
    # Config parameters
    BANKROLL = 50.0 # USDC
    KELLY_K = 0.125
    EV_THRESHOLD = 0.05
    CAP_PCT_BALANCE = 0.15
    CAP_MAX_USDC = 8.0
    
    for market in markets:
        m_id = market["id"]
        if m_id in probs:
            p_model = probs[m_id]
            market_price = market.get("current_odds", 0.5)
            
            # 1. Expected Value (EV) = (p * profit) - (q * loss)
            # profit per 1 USDC bet is (1 - market_price) / market_price * cost? 
            # Actually user formula: EV = (p_model * (1 - market_price)) - ((1 - p_model) * market_price)
            ev = (p_model * (1 - market_price)) - ((1 - p_model) * market_price)
            ev_values[m_id] = round(ev, 4)
            
            # Decide side: if p_model > market_price, buy YES. Else buy NO?
            # Actually use the delta logic
            side = "buy_yes" if p_model > market_price else "buy_no"
            trade_sides[m_id] = side
            
            # Discard trade if EV < 0.05
            if ev < EV_THRESHOLD:
                print(f" -> Skipping {m_id}: EV ({ev:.4f}) below threshold.")
                continue
                
            # 2. Fractional Kelly sizing
            # b = net odds (profit per unit bet). Example: price 0.4 -> b = (1-0.4)/0.4 = 1.5
            if market_price <= 0 or market_price >= 1: continue
            b = (1.0 - market_price) / market_price
            p = p_model
            q = 1.0 - p
            
            f_star = (b * p - q) / b
            
            # Apply conservative k factor
            size_fraction = f_star * KELLY_K
            
            # Convert to absolute USDC
            size_usdc = size_fraction * BANKROLL
            
            # 3. Apply Safety Caps
            # Max 15% of bankroll or $8 per leg
            balance_cap = BANKROLL * CAP_PCT_BALANCE
            final_size = max(0, min(size_usdc, balance_cap, CAP_MAX_USDC))
            
            if final_size > 0.01: # Minimum viable bet
                position_sizes[m_id] = round(final_size, 4)
                print(f" -> Approved trade {m_id}: EV={ev:.4f}, size={final_size:.2f} USDC, side={side}")

    return {
        "ev_values": ev_values,
        "position_sizes": position_sizes,
        "trade_sides": trade_sides,
        "human_approval": True, 
        "cycle_logs": state.get("cycle_logs", []) + [f"Decision: {len(position_sizes)} trades approved with EV filtering and Kelly sizing."]
    }

def risk_guardian_agent(state: AgentState) -> Dict[str, Any]:
    """
    RiskGuardian Node:
    Monitors system safety, VPIN (Volume Probability of Informed Trading) levels,
    and total portfolio exposure. Acts as a safety circuit breaker.
    """
    print("[RISK GUARDIAN] Enforcing VPIN kill switch and exposure caps...")
    
    risk_flags = state.get("risk_flags", [])
    pause_flag = False
    bankroll = 50.0 # From decision_agent config
    
    try:
        poly = pmxt.polymarket()
        
        # 1. VPIN Calculation: |Buy Vol - Sell Vol| / Total Vol
        # We check the most liquid market from the state as a proxy for platform toxicity
        markets = state.get("current_markets", [])
        if markets:
            target_market = sorted(markets, key=lambda x: x.get('liquidity', 0), reverse=True)[0]
            m_id = target_market["id"]
            
            # Fetch recent trades
            trades = poly.fetch_trades(m_id)
            if trades and len(trades) > 5:
                buy_vol = sum(t['amount'] for t in trades if t['side'] == 'buy')
                sell_vol = sum(t['amount'] for t in trades if t['side'] == 'sell')
                total_vol = buy_vol + sell_vol
                
                vpin = abs(buy_vol - sell_vol) / total_vol if total_vol > 0 else 0
                print(f" -> VPIN for {m_id}: {vpin:.4f}")
                
                if vpin > 0.7:
                    risk_flags.append("VPIN_KILL_SWITCH")
                    send_telegram_msg(f"🚨 *KILL SWITCH*: VPIN reached {vpin:.2f} on market {m_id}. Halting execution.")
            else:
                print(" -> Not enough trades to calculate VPIN.")

        # 2. Exposure Monitoring: hits 30% of bankroll
        # In production, fetch actual open positions. Here we track size in state.
        total_exposure = sum(state.get("position_sizes", {}).values())
        if total_exposure > (bankroll * 0.30):
            risk_flags.append("MAX_EXPOSURE_REACHED")
            send_telegram_msg(f"⚠️ *EXPOSURE ALERT*: Total exposure ${total_exposure:.2f} exceeds 30% bucket. Pausing.")

        # 3. Kill-Switch Activation
        if risk_flags:
            poly.cancel_all_orders()
            pause_flag = True
            
        return {
            "risk_flags": risk_flags,
            "pause_flag": pause_flag,
            "total_exposure": total_exposure,
            "cycle_logs": state.get("cycle_logs", []) + [f"RiskGuardian: Checks complete. Flags: {risk_flags}"]
        }
        
    except Exception as e:
        print(f"[ERROR] RiskGuardian failed: {e}")
        return {"risk_flags": state.get("risk_flags", []) + ["GUARDIAN_FETCH_ERROR"]}

def executor_agent(state: AgentState) -> Dict[str, Any]:
    """
    Executor Node:
    Places limit orders on Polymarket via the pmxt SDK.
    Only executes if positive edge is confirmed and no risk flags are raised.
    Uses mid-price + offset for rebate capture and polls for confirmation.
    """
    print("[EXECUTOR] Executing limit orders via pmxt on Polygon...")
    
    pos_sizes = state.get("position_sizes", {})
    trade_sides = state.get("trade_sides", {})
    markets = {m["id"]: m for m in state.get("current_markets", [])}
    
    executed_trades = []
    
    try:
        poly = pmxt.polymarket()
        
        for m_id, size in pos_sizes.items():
            if size <= 0: continue
            
            market = markets.get(m_id)
            if not market: continue
            
            side_type = trade_sides.get(m_id)
            token_id = market["yes_token_id"] if side_type == "buy_yes" else market["no_token_id"]
            
            # 1. Fetch Orderbook to calculate mid price
            orderbook = poly.fetch_order_book(m_id)
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            if not bids or not asks:
                print(f" -> Skipping {m_id}: Orderbook empty.")
                continue
                
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            mid_price = (best_bid + best_ask) / 2
            
            # 2. Set Limit Price (slightly better than mid to capture maker rebates)
            # If buying YES, we want to buy slightly lower than mid if possible, 
            # or exactly at mid/bid to be a maker.
            limit_price = round(mid_price + 0.001, 3) # Aggressive maker
            
            print(f" -> PLACING ORDER: {m_id} | Side: {side_type} | Limit: {limit_price} | Size: {size} USDC")
            
            # 3. Create Order
            if SIMULATION_MODE:
                print(f" -> [SIMULATION] Would place order: {m_id} | Token: {token_id} | Size: {size} USDC | Price: {limit_price}")
                order_id = f"sim_{int(time.time())}_{m_id}"
                status_val = "simulation"
            else:
                if HUMAN_APPROVAL_REQUIRED:
                    # In a real CLI/Bot context, this might wait for input, 
                    # but here we log and skip if not preset or just alert.
                    # For this implementation, we assume human-approval flag in state.
                    if not state.get("human_approval"):
                        print(f" -> [WAITING] Human approval required for {m_id}. Skipping.")
                        continue

                order = poly.create_order(
                    token_id=token_id,
                    side="buy", # All trades here are 'buy' of a specific outcome token
                    size=size,
                    price=limit_price
                )
                order_id = order.get("id")
                status_val = "pending"

            if order_id:
                # 4. Follow-up Poll for Confirmation (Mocked for simulation)
                if not SIMULATION_MODE:
                    time.sleep(2) # Brief wait for matching
                    status = poly.fetch_order(order_id)
                    status_val = status.get('status', 'unknown')
                
                print(f" -> Order {order_id} Status: {status_val}")
                
                db.log_trade(m_id, side_type, size, limit_price, status_val)
                
                executed_trades.append({
                    "market": m_id,
                    "side": side_type,
                    "size": size,
                    "price": limit_price,
                    "status": status_val
                })
        
        # 5. Notify via Telegram
            mode_str = "[SIMULATION]" if SIMULATION_MODE else "✅"
            msg = f"{mode_str} *TRADES EXECUTED*\n"
            for t in executed_trades:
                msg += f"- {t['market']}: {t['side']} {t['size']} USDC @ {t['price']} ({t['status']})\n"
            send_telegram_msg(msg)

        return {
            "cycle_logs": state.get("cycle_logs", []) + [f"Executor: {len(executed_trades)} orders placed."]
        }
        
    except Exception as e:
        print(f"[ERROR] Executor failed: {e}")
        send_telegram_msg(f"❌ *EXECUTOR ERROR*: {str(e)}")
        return {"cycle_logs": state.get("cycle_logs", []) + [f"Executor error: {str(e)}"]}

def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """
    Supervisor Node:
    Orchestrates the entry point and tracks the lifecycle of the autonomy cycle.
    """
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n{'='*60}")
    print(f"AUTONOMOUS CYCLE START: {timestamp}")
    print(f"{'='*60}")
    
    return {
        "cycle_logs": [f"Supervisor: Cycle initialized at {timestamp}."]
    }

# --- 3. Conditional Routing and Graph Construction ---

def route_risk_assessment(state: AgentState) -> str:
    """
    Conditional routing based on RiskGuardian output.
    """
    if state.get("risk_flags"):
        print("!!! HALT: Execution skipped due to Risk Flags !!!")
        return "halt"
    return "proceed"

# Construct the Graph
builder = StateGraph(AgentState)

# Add nodes
builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher", researcher_agent)
builder.add_node("analyst", analyst_agent)
builder.add_node("decision", decision_agent)
builder.add_node("risk_guardian", risk_guardian_agent)
builder.add_node("executor", executor_agent)

# Set workflow edges
builder.set_entry_point("supervisor")
builder.add_edge("supervisor", "researcher")
builder.add_edge("researcher", "analyst")
builder.add_edge("analyst", "decision")
builder.add_edge("decision", "risk_guardian")

# Risk-based conditional edge
builder.add_conditional_edges(
    "risk_guardian",
    route_risk_assessment,
    {
        "proceed": "executor",
        "halt": END
    }
)

builder.add_edge("executor", END)

# Compile the application
app = builder.compile()

# --- 4. Scheduling and Autonomy ---

def run_agent_loop(is_paused: bool = False) -> bool:
    """
    Single iteration of the LangGraph multi-agent flow.
    Returns the pause_flag for the next cycle.
    """
    if is_paused:
        print("[LOOP] Agent is currently PAUSED due to RiskGuardian safety trip.")
        # Optional: logic to check if it's safe to resume
        # For now, we skip execution but check if we should reset
        return True 

    # Initialize state with required fields
    initial_state: AgentState = {
        "current_markets": [],
        "weather_forecasts": {},
        "probabilities": {},
        "edge_deltas": {},
        "ev_values": {},
        "position_sizes": {},
        "trade_sides": {},
        "risk_flags": [],
        "human_approval": False,
        "cycle_logs": [],
        "pause_flag": False,
        "total_exposure": 0.0
    }
    
    try:
        # Run the graph
        final_state = app.invoke(initial_state)
        
        print(f"\nCycle statistics reported by Supervisor:")
        for log in final_state["cycle_logs"]:
            print(f" - {log}")
            
        # Update bankroll history
        # Mocking balance/equity for now
        db.update_bankroll(BANKROLL + pnl, BANKROLL + pnl)

        return final_state.get("pause_flag", False)

    except Exception as e:
        print(f"CRITICAL ERROR in Cycle: {e}")
        send_telegram_msg(f"🚨 *CRITICAL SYSTEM ERROR*: {str(e)}")
        return True # Pause on error for safety

# Main execution loop
if __name__ == "__main__":
    print("--------------------------------------------------")
    print("Weather Arbitrage Autonomous AI Agent v1.0")
    print("Architecture: Multi-Agent LangGraph | Execution: PMXT")
    print(f"Mode: {'SIMULATION' if SIMULATION_MODE else 'LIVE'}")
    print(f"Human Approval: {HUMAN_APPROVAL_REQUIRED}")
    print("--------------------------------------------------")
    
    # Run in a cron-like schedule every 900 seconds
    POLL_INTERVAL = 900 
    system_paused = False
    
    print(f"\n[ACTIVE] Starting autonomy loop. Polling every {POLL_INTERVAL} seconds...")
    while True:
        try:
            system_paused = run_agent_loop(system_paused)
        except KeyboardInterrupt:
            print("\n[STOP] Shutting down agent...")
            break
        except Exception as e:
            print(f"Loop error: {e}")
            system_paused = True # Pause on unexpected loop errors
        
        # Daily Summary logic
        current_hour = datetime.now().hour
        current_minute = datetime.now().minute
        if current_hour == 8 and current_minute < 15: # Run around 8 AM
            summary = db.get_daily_summary()
            summary_msg = f"📅 *DAILY REPORT*\n- Date: {summary['timestamp'][:10]}\n- Trades: {summary['trade_count']}\n- Balance: ${summary['current_balance']:.2f}"
            send_telegram_msg(summary_msg)

        print(f"\n[SLEEP] Waiting {POLL_INTERVAL}s for next cycle...")
        time.sleep(POLL_INTERVAL)
