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
    # New Fields for v2.0
    api_weights: Dict[str, float]
    open_positions: List[Dict[str, Any]]
    bankroll: float

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
                    forecast_link = p_resp.json()["properties"]["forecast"]
                    f_resp = client.get(forecast_link, timeout=10.0)
                    if f_resp.status_code == 200:
                        results["noaa"] = f_resp.json()
    except Exception as e:
        print(f"[ERROR] Weather API call failed: {e}")
    
    return results

def researcher_agent(state: AgentState) -> Dict[str, Any]:
    """
    Researcher Node:
    Fetches active weather markets and live bankroll.
    Includes 'Resolution Source Scraper' to prioritize specific station data.
    """
    print("\n[RESEARCHER] Fetching active weather markets, balance, and source rules...")
    
    try:
        poly = pmxt.polymarket() 
        
        # 1. Fetch Live Bankroll (Dynamic Compounding)
        balance_info = poly.fetch_balance()
        usdc_bal = 50.0 
        if isinstance(balance_info, list):
            for asset in balance_info:
                if asset.get('symbol') == 'USDC':
                    usdc_bal = float(asset.get('free', 50.0))
                    break
        print(f" -> Active Bankroll: ${usdc_bal:.2f} USDC")

        # 2. Fetch Open Positions (for Early Exit logic)
        open_positions = poly.fetch_positions()

        # 3. Fetch Weights from DB
        weights = db.get_api_weights()

        # 4. Fetch Markets and Scrape Resolution Sources
        all_markets = poly.fetch_markets(query="weather")
        weather_markets = []
        unique_locations = set()
        
        for m in all_markets:
            if m.get('active') and m.get('tokens') and m.get('liquidity'):
                desc = m.get('description', '').lower()
                rules = m.get('rules', '').lower()
                
                city_key = "NYC"
                for city in CITY_COORDS.keys():
                    if city.upper() in m.get('question', '').upper():
                        city_key = city
                        break
                
                # Scrape for NOAA station IDs (e.g., KNYC, KLGA)
                station_id = None
                station_match = re.search(r'\b(k[a-z]{3})\b', desc + rules)
                if station_match:
                    station_id = station_match.group(1).upper()
                    print(f" -> Detected specific station resolution: {station_id} for {city_key}")

                market_details = {
                    "id": m.get('id'),
                    "question": m.get('question'),
                    "current_odds": m.get('tokens', [{}])[0].get('price', 0.5), 
                    "liquidity": m.get('liquidity', 0),
                    "yes_token_id": m.get('tokens', [{}, {}])[0].get('token_id'),
                    "no_token_id": m.get('tokens', [{}, {}])[1].get('token_id'),
                    "city": city_key,
                    "station_id": station_id
                }
                
                weather_markets.append(market_details)
                unique_locations.add(city_key)

        print(f" -> Found {len(weather_markets)} markets. Scanned rules for station IDs.")

        # 5. Fetch weather data
        weather_data_map = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_city = {
                executor.submit(fetch_weather_data, CITY_COORDS[c]["lat"], CITY_COORDS[c]["lon"], CITY_COORDS[c]["tz"]): c 
                for c in unique_locations if c in CITY_COORDS
            }
            for future in concurrent.futures.as_completed(future_to_city):
                city = future_to_city[future]
                weather_data_map[city] = future.result()

        return {
            "current_markets": weather_markets,
            "weather_forecasts": weather_data_map,
            "bankroll": usdc_bal,
            "open_positions": open_positions,
            "api_weights": weights,
            "cycle_logs": state.get("cycle_logs", []) + [f"Researcher: {len(weather_markets)} markets. Bankroll: ${usdc_bal}."]
        }
    except Exception as e:
        print(f"[CRITICAL ERROR] Researcher node failed: {e}")
        return {"cycle_logs": state.get("cycle_logs", []) + [f"Researcher error: {str(e)}"]}

def analyst_agent(state: AgentState) -> Dict[str, Any]:
    """
    Analyst Node:
    Processes weighted ensemble forecasts and applies Bayesian updating.
    Calculates the 'fair' probability and edge delta vs Polymarket pricing.
    """
    print("[ANALYST] Running weighted ensemble models and Bayesian updates...")
    
    p_models = {}
    edge_deltas = {}
    api_weights = state.get("api_weights", {"open_meteo": 1.0, "noaa": 1.2})
    
    markets = state.get("current_markets", [])
    forecasts = state.get("weather_forecasts", {})
    
    for market in markets:
        m_id = market["id"]
        question = market["question"]
        city = market.get("city")
        
        # 1. Extract Thresholds
        upper_bound = float('inf')
        lower_bound = float('-inf')
        bucket_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:-|to|and)\s*(\d+(?:\.\d+)?)', question)
        if bucket_match:
            lower_bound = float(bucket_match.group(1))
            upper_bound = float(bucket_match.group(2))
        else:
            match_above = re.search(r'(?:above|over|>|greater than)\s+(\d+(?:\.\d+)?)\s*(?:f\b|degrees)?', question, re.I)
            match_below = re.search(r'(?:below|under|<|less than)\s+(\d+(?:\.\d+)?)\s*(?:f\b|degrees)?', question, re.I)
            if match_above:
                lower_bound = float(match_above.group(1))
            elif match_below:
                upper_bound = float(match_below.group(1))

        # 2. Weighted Ensemble Calculation
        city_data = forecasts.get(city, {})
        om_temps = city_data.get("open_meteo", {}).get("hourly", {}).get("temperature_2m", [])
        
        # Calculate ensemble mu and sigma
        samples = []
        if om_temps:
            w = api_weights.get("open_meteo", 1.0)
            samples.extend([t for t in om_temps[:24]] * int(w * 10))
            
        if not samples:
            print(f" -> Skipping {m_id}: No ensemble data for {city}")
            continue

        mu = np.mean(samples)
        sigma = np.std(samples)
        if sigma < 0.1: sigma = 0.5 

        # 3. Model Probability (Likelihood) using CDF
        z_upper = (upper_bound - mu) / sigma
        z_lower = (lower_bound - mu) / sigma
        likelihood = stats.norm.cdf(z_upper) - stats.norm.cdf(z_lower)
        likelihood = max(0.0001, min(0.9999, likelihood))

        # 4. Bayesian Updating
        prior = db.get_prior(m_id)
        evidence = (likelihood * prior) + ((1 - likelihood) * (1 - prior))
        p_model = (likelihood * prior) / evidence if evidence > 0 else likelihood
        
        db.save_prior(m_id, float(p_model))
        
        # 5. Edge Detection
        price_yes = market.get("current_odds", 0.5)
        p_market = price_yes
        
        delta = p_model - p_market
        
        p_models[m_id] = round(float(p_model), 4)
        edge_deltas[m_id] = round(float(delta), 4)
        
        if abs(delta) > 0.05:
            print(f" -> Market: {m_id} | Post: {p_model:.2f} | Mkt: {p_market:.2f} | Edge: {delta:+.2f}")

    return {
        "probabilities": p_models,
        "edge_deltas": edge_deltas,
        "cycle_logs": state.get("cycle_logs", []) + [f"Analyst: Refined {len(p_models)} probabilities."]
    }

def decision_agent(state: AgentState) -> Dict[str, Any]:
    """
    Decision Node:
    Applies Expected Value (EV), Kelly sizing, and Early Exit logic.
    """
    print("[DECISION] Running EV filtering, Kelly sizing, and Early Exit check...")
    
    probs = state["probabilities"]
    markets = state.get("current_markets", [])
    open_positions = state.get("open_positions", [])
    bankroll = state.get("bankroll", 50.0) 
    
    ev_values, position_sizes, trade_sides = {}, {}, {}
    
    # 1. Early Exit Logic (Capital Recycling)
    for pos in open_positions:
        m_id = pos.get('market_id')
        if m_id in probs:
            p_model = probs[m_id]
            current_price = pos.get('current_price', 0.5)
            # If price reached model expectation (within 1.5%), exit to recycle capital
            if abs(p_model - current_price) < 0.015:
                print(f" -> EARLY EXIT triggered for {m_id}")
                trade_sides[f"EXIT_{m_id}"] = "sell"
                position_sizes[f"EXIT_{m_id}"] = pos.get('size', 0)

    # 2. New Trade Identification & Kelly Sizing (Compounding)
    for market in markets:
        m_id = market["id"]
        if m_id in probs:
            p_model = probs[m_id]
            price_yes = market.get("current_odds", 0.5)
            
            # EV Calculation
            ev = (p_model * (1 - price_yes)) - ((1 - p_model) * price_yes)
            
            if ev > 0.05:
                b = (1.0 - price_yes) / price_yes
                f_star = (b * p_model - (1 - p_model)) / b
                size_usdc = f_star * 0.125 * bankroll
                final_size = max(0, min(size_usdc, bankroll * 0.15, 8.0))
                
                if final_size > 0.5:
                    ev_values[m_id] = round(ev, 4)
                    position_sizes[m_id] = round(final_size, 4)
                    trade_sides[m_id] = "buy_yes" if p_model > price_yes else "buy_no"
                    print(f" -> Approved: {m_id} | EV: {ev:.2f} | Size: {final_size:.2f}")

    return {
        "ev_values": ev_values,
        "position_sizes": position_sizes,
        "trade_sides": trade_sides,
        "human_approval": True, 
        "cycle_logs": state.get("cycle_logs", []) + [f"Decision: {len(position_sizes)} trades/exits."]
    }

def risk_guardian_agent(state: AgentState) -> Dict[str, Any]:
    """
    RiskGuardian Node:
    Monitors system safety, VPIN levels, and total portfolio exposure.
    """
    print("[RISK GUARDIAN] Enforcing VPIN kill switch and exposure caps...")
    
    risk_flags = state.get("risk_flags", [])
    pause_flag = False
    bankroll = state.get("bankroll", 50.0)
    
    try:
        poly = pmxt.polymarket()
        
        # 1. VPIN Calculation with Sensitivity Tuning
        markets = state.get("current_markets", [])
        if markets:
            target_market = sorted(markets, key=lambda x: x.get('liquidity', 0), reverse=True)[0]
            m_id = target_market["id"]
            
            trades = poly.fetch_trades(m_id)
            if trades and len(trades) > 5:
                buy_vol = sum(t['amount'] for t in trades if t['side'] == 'buy')
                sell_vol = sum(t['amount'] for t in trades if t['side'] == 'sell')
                total_vol = buy_vol + sell_vol
                vpin = abs(buy_vol - sell_vol) / total_vol if total_vol > 0 else 0
                
                # SENSITIVITY TUNING: 0.5 threshold instead of 0.7 during 'toxic' periods
                threshold = 0.5 if "hour_to_close" in m_id else 0.7 
                
                if vpin > threshold:
                    risk_flags.append("VPIN_KILL_SWITCH")
                    send_telegram_msg(f"🚨 *KILL SWITCH*: VPIN {vpin:.2f} > {threshold}. Halting.")

        # 2. Exposure Monitoring
        total_exposure = sum(state.get("position_sizes", {}).values())
        if total_exposure > (bankroll * 0.35): # Slightly higher bucket allowed with compounding
            risk_flags.append("MAX_EXPOSURE_REACHED")
            send_telegram_msg(f"⚠️ *EXPOSURE ALERT*: Total exposure ${total_exposure:.2f} exceeds 35% limit.")

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
    Places 'Maker-Only' limit orders via pmxt. 
    Handles both new entries and 'Early Exit' sell orders.
    """
    print("[EXECUTOR] Executing maker-only limit orders...")
    
    pos_sizes = state.get("position_sizes", {})
    trade_sides = state.get("trade_sides", {})
    markets = {m["id"]: m for m in state.get("current_markets", [])}
    
    executed_trades = []
    
    try:
        poly = pmxt.polymarket()
        
        for m_id, size in pos_sizes.items():
            is_exit = m_id.startswith("EXIT_")
            real_m_id = m_id.replace("EXIT_", "") if is_exit else m_id
            
            market = markets.get(real_m_id)
            if not market: continue
            
            side_type = trade_sides.get(m_id)
            token_id = market["yes_token_id"] if "yes" in str(side_type) else market["no_token_id"]
            
            # 1. Fetch Orderbook for Maker price
            ob = poly.fetch_order_book(real_m_id)
            bids = ob.get('bids', [])
            asks = ob.get('asks', [])
            
            if not bids or not asks: continue
            
            # Maker Price Logic: Aggressive maker (1 tick better than best)
            if "buy" in str(side_type):
                limit_price = bids[0][0] + 0.001
                order_side = "buy"
            else:
                limit_price = asks[0][0] - 0.001
                order_side = "sell"

            if SIMULATION_MODE:
                print(f" -> [SIMULATION] Maker Order: {real_m_id} | {order_side} | Price: {limit_price}")
                executed_trades.append({"market": real_m_id, "side": order_side, "size": size, "price": limit_price, "status": "simulation"})
            else:
                order = poly.create_order(
                    token_id=token_id,
                    side=order_side,
                    size=size,
                    price=limit_price,
                    post_only=True # ENSURE MAKER ONLY
                )
                executed_trades.append({"market": real_m_id, "side": order_side, "size": size, "price": limit_price, "status": order.get('status')})
                db.log_trade(real_m_id, order_side, size, limit_price, order.get('status'))

        # Notify via Telegram
        if executed_trades:
            msg = "✅ *TRADES EXECUTED (v2.0 Maker-Only)*\n"
            for t in executed_trades:
                msg += f"- {t['market']}: {t['side']} {t['size']} USDC @ {t['price']} ({t['status']})\n"
            send_telegram_msg(msg)

        return {
            "cycle_logs": state.get("cycle_logs", []) + [f"Executor: {len(executed_trades)} maker orders dispatched."]
        }
        
    except Exception as e:
        print(f"[ERROR] Executor failed: {e}")
        return {"cycle_logs": state.get("cycle_logs", []) + [f"Executor error: {str(e)}"]}

def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """
    Supervisor Node:
    Orchestrates the lifecycle of the autonomy cycle.
    """
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n{'='*60}")
    print(f"AUTONOMOUS CYCLE START (v2.0): {timestamp}")
    print(f"{'='*60}")
    
    return {
        "cycle_logs": [f"Supervisor: Cycle initialized at {timestamp}."]
    }

# --- Graph Construction ---

def route_risk_assessment(state: AgentState) -> str:
    if state.get("risk_flags"):
        print("!!! HALT: Execution skipped due to Risk Flags !!!")
        return "halt"
    return "proceed"

builder = StateGraph(AgentState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher", researcher_agent)
builder.add_node("analyst", analyst_agent)
builder.add_node("decision", decision_agent)
builder.add_node("risk_guardian", risk_guardian_agent)
builder.add_node("executor", executor_agent)

builder.set_entry_point("supervisor")
builder.add_edge("supervisor", "researcher")
builder.add_edge("researcher", "analyst")
builder.add_edge("analyst", "decision")
builder.add_edge("decision", "risk_guardian")
builder.add_conditional_edges("risk_guardian", route_risk_assessment, {"proceed": "executor", "halt": END})
builder.add_edge("executor", END)

app = builder.compile()

# --- Scheduling and Autonomy ---

def run_agent_loop(is_paused: bool = False) -> bool:
    if is_paused:
        print("[LOOP] Agent is currently PAUSED.")
        return True 

    initial_state: AgentState = {
        "current_markets": [], "weather_forecasts": {}, "probabilities": {},
        "edge_deltas": {}, "ev_values": {}, "position_sizes": {}, "trade_sides": {},
        "risk_flags": [], "human_approval": False, "cycle_logs": [], "pause_flag": False,
        "total_exposure": 0.0, "api_weights": {}, "open_positions": [], "bankroll": 50.0
    }
    
    try:
        final_state = app.invoke(initial_state)
        for log in final_state["cycle_logs"]:
            print(f" - {log}")
        
        # Update bankroll history
        db.update_bankroll(final_state.get("bankroll", 50.0), final_state.get("bankroll", 50.0))
        return final_state.get("pause_flag", False)

    except Exception as e:
        print(f"CRITICAL ERROR in Cycle: {e}")
        return True 

if __name__ == "__main__":
    print("--------------------------------------------------")
    print("Weather Arbitrage Autonomous AI Agent v2.0 (High Velocity)")
    print(f"Mode: {'SIMULATION' if SIMULATION_MODE else 'LIVE'}")
    print("--------------------------------------------------")
    
    POLL_INTERVAL = 60 
    system_paused = False
    
    while True:
        try:
            system_paused = run_agent_loop(system_paused)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Loop error: {e}")
            system_paused = True 
        
        time.sleep(POLL_INTERVAL)
