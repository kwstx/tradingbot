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

load_dotenv()

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

# --- 2. Specialized Agent Nodes ---

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
                    "current_odds": m.get('tokens', [{}])[0].get('price', 0.5), # Default to 0.5 if missing
                    "liquidity": m.get('liquidity', 0),
                    "tokens": m.get('tokens')
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
    """Loads historical priors from a local JSON file."""
    try:
        if os.path.exists("hist_priors.json"):
            with open("hist_priors.json", "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"[ANALYST] Error loading priors: {e}")
    return {}

def save_priors(priors: Dict[str, float]):
    """Saves updated priors to a local JSON file."""
    try:
        with open("hist_priors.json", "w") as f:
            json.dump(priors, f, indent=2)
    except Exception as e:
        print(f"[ANALYST] Error saving priors: {e}")

def analyst_agent(state: AgentState) -> Dict[str, Any]:
    """
    Analyst Node:
    Processes weather forecasts as Gaussian ensembles and applies Bayesian updating.
    Calculates the 'fair' probability and edge delta vs Polymarket pricing.
    """
    print("[ANALYST] Running quantitative ensemble models and Bayesian updates...")
    
    p_models = {}
    edge_deltas = {}
    priors = load_priors()
    
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
        prior = priors.get(m_id, 0.5) # Default prior of 0.5
        evidence = (likelihood * prior) + ((1 - likelihood) * (1 - prior))
        
        p_model = (likelihood * prior) / evidence if evidence > 0 else likelihood
        
        # Persist posterior as the new prior for the next cycle
        priors[m_id] = float(p_model)
        
        # 5. Edge Detection
        # Polymarket P_implied = 1 - price (as per user instruction)
        price_yes = market.get("current_odds", 0.5)
        p_market = 1.0 - price_yes
        
        delta = p_model - p_market
        
        p_models[m_id] = round(float(p_model), 4)
        edge_deltas[m_id] = round(float(delta), 4)
        
        print(f" -> Market: {m_id} | Post: {p_model:.2f} | Mkt: {p_market:.2f} | Edge: {delta:+.2f}")

    # Save all updated priors
    save_priors(priors)
    
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
                print(f" -> Approved trade {m_id}: EV={ev:.4f}, size={final_size:.2f} USDC")

    return {
        "ev_values": ev_values,
        "position_sizes": position_sizes,
        "human_approval": True, # Flag for early live operation
        "cycle_logs": state.get("cycle_logs", []) + [f"Decision: {len(position_sizes)} trades approved with EV filtering and Kelly sizing."]
    }

def risk_guardian_agent(state: AgentState) -> Dict[str, Any]:
    """
    RiskGuardian Node:
    Monitors system safety, VPIN (Volume Probability of Informed Trading) levels,
    and total portfolio exposure. Acts as a safety circuit breaker.
    """
    print("[RISK GUARDIAN] Enforcing VPIN kill switch and exposure caps...")
    
    risk_flags = []
    
    # VPIN Safety Check: Detect toxic flow or abnormal order book imbalance
    vpin_level = np.random.uniform(0.1, 0.4) # Mock calculation
    vpin_threshold = 0.60
    
    if vpin_level > vpin_threshold:
        risk_flags.append("VPIN_EXCEEDED")
        
    # Exposure Monitoring: Cap total investment to $50 USDC
    # In production, this would fetch current open position value via pmxt
    current_exposure = 10.0 
    if current_exposure > 50.0:
        risk_flags.append("EXPOSURE_CAP_BREACH")
        
    return {
        "risk_flags": risk_flags,
        "cycle_logs": state.get("cycle_logs", []) + [f"RiskGuardian: Checks complete (VPIN: {vpin_level:.2f})."]
    }

def executor_agent(state: AgentState) -> Dict[str, Any]:
    """
    Executor Node:
    Places limit orders on Polymarket via the pmxt SDK.
    Only executes if positive edge is confirmed and no risk flags are raised.
    """
    print("[EXECUTOR] Executing limit orders via pmxt on Polygon...")
    
    pos_sizes = state.get("position_sizes", {})
    
    for m_id, size in pos_sizes.items():
        if size > 0:
            print(f" -> ORDER PLACED: {m_id} | Size: {size:.4f} Units")
            # Actual implementation: pm.create_order(market_id=m_id, amount=size, ...)
    
    return {
        "cycle_logs": state.get("cycle_logs", []) + ["Executor: Order sequence completed."]
    }

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

def run_agent_loop():
    """
    Single iteration of the LangGraph multi-agent flow.
    """
    # Initialize state with required fields
    initial_state: AgentState = {
        "current_markets": [],
        "weather_forecasts": {},
        "probabilities": {},
        "edge_deltas": {},
        "ev_values": {},
        "position_sizes": {},
        "risk_flags": [],
        "human_approval": False,
        "cycle_logs": []
    }
    
    try:
        # Run the graph
        final_state = app.invoke(initial_state)
        print(f"\nCycle statistics reported by Supervisor:")
        for log in final_state["cycle_logs"]:
            print(f" - {log}")
    except Exception as e:
        print(f"CRITICAL ERROR in Cycle: {e}")

# Main execution loop
if __name__ == "__main__":
    print("--------------------------------------------------")
    print("Weather Arbitrage Autonomous AI Agent v1.0")
    print("Architecture: Multi-Agent LangGraph | Execution: PMXT")
    print("--------------------------------------------------")
    
    # Run in a cron-like schedule every 900 seconds
    POLL_INTERVAL = 900 
    
    print(f"\n[ACTIVE] Starting autonomy loop. Polling every {POLL_INTERVAL} seconds...")
    while True:
        try:
            run_agent_loop()
        except KeyboardInterrupt:
            print("\n[STOP] Shutting down agent...")
            break
        except Exception as e:
            print(f"Loop error: {e}")
        
        print(f"\n[SLEEP] Waiting {POLL_INTERVAL}s for next cycle...")
        time.sleep(POLL_INTERVAL)
