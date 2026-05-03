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
from datetime import datetime, timedelta

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
from source_scraper import ResolutionSourceScraper
from reliability import ReliabilityManager

load_dotenv()

# --- Simulation & Paper Trading Flags ---
SIMULATION_MODE = os.getenv("SIMULATION_MODE", "false").lower() == "true"
PAPER_TRADING_MODE = os.getenv("PAPER_TRADING_MODE", "true").lower() == "true"
HUMAN_APPROVAL_REQUIRED = os.getenv("HUMAN_APPROVAL_REQUIRED", "false").lower() == "true"

# Initialize Persistence
db = PersistenceManager()

# --- 0. Global Config ---
DEFAULT_BANKROLL = 50.0
MAX_POSITION_SIZE_PCT = 0.15 # Reduced for better risk adjusted returns
KELLY_FRACTION = 0.40      # Reduced to 0.40 to minimize variance and noise
MAX_LIQUIDITY_TAKE_PCT = 0.30 
STALE_PRICE_THRESHOLD_MINS = 30 
MIN_TIME_TO_RESOLUTION_HOURS = 4 

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
    in_flight_exposure: float
    forecast_metadata: Dict[str, Dict[str, float]]

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
        poly = pmxt.Polymarket() 
        
        # 1. Fetch Bankroll (Paper vs Live)
        if PAPER_TRADING_MODE:
            usdc_bal = db.get_latest_bankroll(mode='PAPER', default=DEFAULT_BANKROLL)
            print(f" -> Paper Bankroll: ${usdc_bal:.2f} USDC")
        else:
            usdc_bal = DEFAULT_BANKROLL 
            try:
                balance_info = poly.fetch_balance()
                if isinstance(balance_info, list):
                    for asset in balance_info:
                        if asset.get('symbol') == 'USDC':
                            usdc_bal = float(asset.get('free', DEFAULT_BANKROLL))
                            break
                print(f" -> Active Bankroll: ${usdc_bal:.2f} USDC")
            except Exception as e:
                print(f" -> [WARN] Balance fetch failed: {e}. Using fallback.")

        # 2. Fetch Open Positions (for Early Exit logic)
        if PAPER_TRADING_MODE:
            open_positions = []
        else:
            try:
                open_positions = poly.fetch_positions()
            except Exception as e:
                print(f" -> [WARN] Position fetch failed: {e}. Using empty list.")
                open_positions = []

        # 3. Fetch Weights from DB
        weights = db.get_api_weights()

        # 4. Fetch Markets and Scrape Resolution Sources
        all_markets = poly.fetch_markets(query="weather")
        weather_markets = []
        unique_locations = set()
        
        for m in all_markets:
            # Handle both object-style (new PMXT) and dict-style (old PMXT)
            try:
                is_active = getattr(m, 'status', None) == 'active' or getattr(m, 'active', False)
                has_liquidity = getattr(m, 'liquidity', 0) > 0
                
                if is_active and has_liquidity:
                    question = getattr(m, 'question', getattr(m, 'title', ''))
                    desc = getattr(m, 'description', '')
                    
                    # Extract prices and token IDs
                    if hasattr(m, 'yes'):
                        price_yes = getattr(m.yes, 'price', 0.5)
                        price_no = getattr(m.no, 'price', 0.5)
                        yes_token_id = getattr(m.yes, 'outcome_id', None)
                        no_token_id = getattr(m.no, 'outcome_id', None)
                    else:
                        # Fallback for dict-style
                        tokens = getattr(m, 'tokens', m.get('tokens', [{}, {}]) if hasattr(m, 'get') else [{}, {}])
                        price_yes = float(tokens[0].get('price', 0.5)) if hasattr(tokens[0], 'get') else 0.5
                        price_no = float(tokens[1].get('price', 0.5)) if hasattr(tokens[1], 'get') else 0.5
                        yes_token_id = tokens[0].get('token_id') if hasattr(tokens[0], 'get') else None
                        no_token_id = tokens[1].get('token_id') if hasattr(tokens[1], 'get') else None
                    
                    m_id = getattr(m, 'market_id', getattr(m, 'id', None))
                    res_time = getattr(m, 'resolution_date', getattr(m, 'end_date_iso', None))
                    
                    # 1. Fetch Order Book Depth for Liquidity & Slippage
                    try:
                        # We use the yes_token_id for the primary liquidity check
                        # In PMXT, token_ids are available on the market outcomes
                        primary_token_id = yes_token_id if yes_token_id else (tokens[0].get('token_id') if tokens else None)
                        if primary_token_id:
                            ob = poly.fetch_order_book(primary_token_id)
                            # Sum volume at top 3 levels for a more realistic liquidity picture
                            bids = getattr(ob, 'bids', [])[:3]
                            asks = getattr(ob, 'asks', [])[:3]
                            bid_liquidity = sum(float(getattr(b, 'amount', 0)) for b in bids)
                            ask_liquidity = sum(float(getattr(a, 'amount', 0)) for a in asks)
                        else:
                            bid_liquidity, ask_liquidity = 0, 0
                    except Exception as e:
                        print(f" -> [WARN] Liquidity fetch failed for {m_id}: {e}")
                        bid_liquidity, ask_liquidity = 0, 0

                    city_key = "NYC"
                    for city in CITY_COORDS.keys():
                        if city.upper() in question.upper():
                            city_key = city
                            break
                    
                    # NEW: Use Resolution Source Scraper for high-precision targeting
                    scraper_results = ResolutionSourceScraper.scrape(desc, "")
                    station_id = scraper_results["station_id"]
                    
                    # Refine coordinates if a specific station was found
                    market_lat, market_lon = CITY_COORDS[city_key]["lat"], CITY_COORDS[city_key]["lon"]
                    if scraper_results["coordinates"]:
                        market_lat, market_lon = scraper_results["coordinates"]
                        print(f" -> REFINED COORDS: Using {station_id} coordinates for {city_key}")

                    market_details = {
                        "id": m_id,
                        "question": question,
                        "price_yes": price_yes,
                        "price_no": price_no,
                        "liquidity": getattr(m, 'liquidity', 0),
                        "bid_liquidity": bid_liquidity,
                        "ask_liquidity": ask_liquidity,
                        "yes_token_id": yes_token_id,
                        "no_token_id": no_token_id,
                        "city": city_key,
                        "station_id": station_id,
                        "lat": market_lat,
                        "lon": market_lon,
                        "tz": CITY_COORDS[city_key]["tz"],
                        "resolution_time": res_time
                    }
                    
                    weather_markets.append(market_details)
                    unique_locations.add(city_key)
            except Exception as e:
                print(f" -> [WARN] Skipping market due to parse error: {e}")
                continue

        print(f" -> Found {len(weather_markets)} markets. Scanned rules for station IDs.")

        # 5. Fetch weather data (using specific station coordinates if available)
        weather_data_map = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            unique_targets = {}
            for m in weather_markets:
                target_key = f"{m['lat']}_{m['lon']}"
                unique_targets[target_key] = (m['lat'], m['lon'], m['tz'])

            future_to_key = {
                executor.submit(fetch_weather_data, lat, lon, tz): key 
                for key, (lat, lon, tz) in unique_targets.items()
            }
            
            temp_weather_results = {}
            for future in concurrent.futures.as_completed(future_to_key):
                key = future_to_key[future]
                res = future.result()
                # Add timestamp to weather data for freshness check
                if res:
                    res["fetched_at"] = datetime.now().isoformat()
                temp_weather_results[key] = res
            
            # Map back to market-friendly format
            for m in weather_markets:
                target_key = f"{m['lat']}_{m['lon']}"
                weather_data_map[m['id']] = temp_weather_results.get(target_key)

        # 5. Fetch Pending Exposure (Locked Capital)
        in_flight = db.get_total_paper_exposure() if PAPER_TRADING_MODE else 0.0

        return {
            "current_markets": weather_markets,
            "weather_forecasts": weather_data_map,
            "bankroll": usdc_bal,
            "in_flight_exposure": in_flight,
            "open_positions": open_positions,
            "api_weights": weights,
            "cycle_logs": state.get("cycle_logs", []) + [f"Researcher: {len(weather_markets)} markets. Bankroll: ${usdc_bal}. In-Flight: ${in_flight}."]
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
        
        # 0. Data Freshness & Leakage Protection
        res_time_str = market.get("resolution_time")
        if res_time_str:
            try:
                res_time_str = str(res_time_str).replace('Z', '+00:00')
                res_dt = datetime.fromisoformat(res_time_str)
                now = datetime.now(res_dt.tzinfo) if res_dt.tzinfo else datetime.now()
                hours_to_res = (res_dt - now).total_seconds() / 3600
                
                if hours_to_res < MIN_TIME_TO_RESOLUTION_HOURS:
                    print(f" -> [PROTECT] Skipping {m_id}: Resolution too close ({hours_to_res:.1f}h). Avoiding leakage.")
                    continue
            except: pass

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
        city_data = forecasts.get(m_id, {})
        om_data = city_data.get("open_meteo", {})
        noaa_data = city_data.get("noaa", {})
        
        # Target Date Extraction (Fallback to tomorrow if not found)
        target_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', question)
        if date_match:
            target_date = date_match.group(1)

        api_preds = []
        # Process individual API forecasts
        for api_name, data in [("open_meteo", om_data), ("noaa", noaa_data)]:
            if not data: continue
            
            api_temps = []
            if api_name == "open_meteo":
                api_temps = data.get("hourly", {}).get("temperature_2m", [])[:24]
            elif api_name == "noaa":
                api_temps = [p.get("temperature") for p in data.get("properties", {}).get("periods", []) if p.get("temperature") is not None][:12]
            
            if api_temps:
                # CRITICAL FIX: Most markets are about the HIGH or LOW of the day.
                # If the question is 'above X', we care about the MAX temperature.
                # If the question is 'below X', we care about the MIN temperature.
                if lower_bound != float('-inf') and upper_bound == float('inf'):
                    pred = np.max(api_temps)
                elif upper_bound != float('inf') and lower_bound == float('-inf'):
                    pred = np.min(api_temps)
                else:
                    pred = np.mean(api_temps) # Range/Bucket questions
                
                # Weighting
                w = api_weights.get(api_name, 1.0)
                api_preds.extend([pred] * int(w * 10))

                # Save to History for Reliability Tracking
                threshold = lower_bound if lower_bound != float('-inf') else upper_bound
                db.save_forecast(m_id, api_name, float(np.mean(api_temps)), float(np.std(api_temps)), threshold, target_date, market["lat"], market["lon"])

        if not api_preds:
            print(f" -> Skipping {m_id}: No ensemble data available.")
            continue

        mu = np.mean(api_preds)
        
        # 3. Model Probability (Likelihood) using Volatility-Adjusted SEM
        # We adjust the baseline sigma (2.0) by the disagreement between APIs
        disagreement = np.std(list(set(api_preds))) if len(set(api_preds)) > 1 else 0.5
        adjusted_sigma = 2.0 + disagreement
        sem = adjusted_sigma / np.sqrt(len(api_preds))
        
        z_upper = (upper_bound - mu) / sem
        z_lower = (lower_bound - mu) / sem
        likelihood = stats.norm.cdf(z_upper) - stats.norm.cdf(z_lower)
        likelihood = max(0.0001, min(0.9999, likelihood))

        # 4. Bayesian Updating & Calibration
        prior = db.get_prior(m_id)
        evidence = (likelihood * prior) + ((1 - likelihood) * (1 - prior))
        unscaled_posterior = (likelihood * prior) / evidence if evidence > 0 else likelihood
        p_model = prior + 0.7 * (unscaled_posterior - prior) # Slightly more dampening
        
        # PRO-LEVEL: Tail-Focused Calibration
        # Deep power transform to ignore everything except high-conviction mispricings.
        p_model = p_model**1.5 if p_model < 0.5 else 1 - (1 - p_model)**1.5

        db.save_prior(m_id, float(p_model))
        
        # 5. Edge Detection (Initial Delta vs YES for logging)
        price_yes = market.get("price_yes", 0.5)
        p_market = price_yes
        
        delta = p_model - p_market
        
        p_models[m_id] = round(float(p_model), 4)
        if "forecast_metadata" not in state: state["forecast_metadata"] = {}
        state["forecast_metadata"][m_id] = {
            "mu": float(mu), 
            "sem": float(sem),
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }
        
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
    Implemented 'Capital Recycler' to boost velocity when high-edge trades (10%+) exist.
    """
    print("[DECISION] Running EV filtering, Kelly sizing, and Early Exit check...")
    
    probs = state["probabilities"]
    edge_deltas = state.get("edge_deltas", {})
    markets = state.get("current_markets", [])
    open_positions = state.get("open_positions", [])
    bankroll = state.get("bankroll", DEFAULT_BANKROLL) 
    in_flight = state.get("in_flight_exposure", 0.0)
    
    # Capital Hardening: Available balance is total bankroll minus already locked capital
    # We also enforce a 35% total exposure cap across all trades
    max_total_allowed = bankroll * 0.35
    running_exposure = in_flight
    remaining_to_allocate = max(0, max_total_allowed - in_flight)
    
    ev_values, position_sizes, trade_sides = {}, {}, {}
    
    # 1. Identify High-Edge Opportunities (Capital Recycler trigger)
    # We look for ANY market where the edge is > 10% to prioritize capital velocity
    high_edge_available = any(abs(delta) > 0.10 for delta in edge_deltas.values())
    
    # 2. Early Exit & Capital Recycler Logic
    for pos in open_positions:
        m_id = pos.get('market_id')
        if m_id in probs:
            p_model = probs[m_id]
            current_price = pos.get('current_price', 0.5)
            
            # Logic: If price is within $0.05 of our model value, and we have >$0.90 locked,
            # or if we're within 1.5% and just want to take profit.
            near_target = abs(p_model - current_price) < 0.05
            high_value_locked = current_price > 0.90
            
            # Trigger recycler if we have a great new trade waiting
            if near_target and high_value_locked and high_edge_available:
                print(f" -> VELOCITY RECYCLER: Selling {m_id} early to capture 10%+ edge elsewhere.")
                trade_sides[f"EXIT_{m_id}"] = f"sell_{pos.get('side', 'yes')}" # Track side
                position_sizes[f"EXIT_{m_id}"] = pos.get('size', 0)
            
            # Standard profit taking disabled to maximize resolution payout in backtests
            # elif abs(p_model - current_price) < 0.015:
            #    print(f" -> PROFIT TAKING: Early exit for {m_id} (Target reached).")
            #    trade_sides[f"EXIT_{m_id}"] = f"sell_{pos.get('side', 'yes')}"
            #    position_sizes[f"EXIT_{m_id}"] = pos.get('size', 0)

    # 3. Symbiotic Arb: Dual-Token Price Comparison
    for market in markets:
        m_id = market["id"]
        if m_id in probs:
            p_yes_model = probs[m_id]
            p_no_model = 1.0 - p_yes_model
            
            price_yes = market.get("price_yes", 0.5)
            price_no = market.get("price_no", 0.5)
            
            # Mathematical Yield (EV) for both sides
            # EV = (WinProb * Profit) - (LossProb * Loss)
            # Profit on $1 is (1/Price - 1). Loss is $1.
            # Simplified EV per $1: (WinProb / Price) - 1
            ev_yes = (p_yes_model / price_yes) - 1 if price_yes > 0 else -1
            ev_no = (p_no_model / price_no) - 1 if price_no > 0 else -1
            
            # Mathematical Yield (EV) - Fee Aware
            # Using a target size of $5 for EV estimation on this small bankroll
            est_fixed_fee_impact = 0.05 / 5.0 
            pct_fee = 0.001
            
            ev_yes = (p_yes_model / price_yes) - 1 - est_fixed_fee_impact - pct_fee if price_yes > 0 else -1
            ev_no = (p_no_model / price_no) - 1 - est_fixed_fee_impact - pct_fee if price_no > 0 else -1
            
            # HIGH-CONVICTION THRESHOLD: EV > 8% and edge > 12%
            edge_yes = p_yes_model - price_yes
            edge_no = p_no_model - price_no
            
            is_uncertain = 0.38 < p_yes_model < 0.62 

            if not is_uncertain and ev_yes > ev_no and ev_yes > 0.08 and edge_yes > 0.12:
                best_ev = ev_yes
                side = "buy_yes"
                p_win = p_yes_model
                price_entry = price_yes
            elif not is_uncertain and ev_no > ev_yes and ev_no > 0.10 and edge_no > 0.15:
                best_ev = ev_no
                side = "buy_no"
                p_win = p_no_model
                price_entry = price_no
            else:
                continue # No significant edge on either side

            # Kelly Sizing with Confidence Scaling (Z-Score)
            # We scale the Kelly fraction by how 'extreme' our prediction is
            meta = state.get("forecast_metadata", {}).get(m_id, {})
            mu_val = meta.get("mu", 0)
            sem_val = meta.get("sem", 1.0)
            l_bound = meta.get("lower_bound", float('-inf'))
            u_bound = meta.get("upper_bound", float('inf'))
            
            z_conf = abs(mu_val - (l_bound if l_bound != float('-inf') else u_bound)) / sem_val
            # Sharpened scaling: Requires 4-sigma for full size
            conf_multiplier = min(1.0, z_conf / 4.0) 
            
            # f* = (bp - q) / b 
            b = (1.0 / price_entry) - 1
            f_star = (b * p_win - (1 - p_win)) / b if b > 0 else 0
            
            size_usdc = f_star * KELLY_FRACTION * conf_multiplier * bankroll
            
            # --- REAL-WORLD HARDENING ---
            # 1. Liquidity Cap: Don't take more than X% of available depth
            available_liquidity = market.get("ask_liquidity" if "buy" in side else "bid_liquidity", 0)
            if available_liquidity > 0:
                liquidity_limit = available_liquidity * MAX_LIQUIDITY_TAKE_PCT
                size_usdc = min(size_usdc, liquidity_limit)
                if size_usdc < liquidity_limit * 0.1: # If our calculated size is tiny, ignore it
                    pass 
            else:
                # If no order book data, use a very small fixed cap
                size_usdc = min(size_usdc, 5.0)

            # 2. Dynamic Scaling: Allow sizes to scale with bankroll, capped at 10% per trade
            # AND capped by the remaining total exposure allowed for this cycle
            final_size = max(0, min(size_usdc, bankroll * MAX_POSITION_SIZE_PCT, remaining_to_allocate))
            
            # Fee Guard: Minimum size to ensure fees stay under 1% for $50 bankroll
            if final_size > 5.0: 
                ev_values[m_id] = round(best_ev, 4)
                position_sizes[m_id] = round(final_size, 4)
                trade_sides[m_id] = side
                
                # Update running allocation trackers
                remaining_to_allocate -= final_size
                running_exposure += final_size
                
                print(f" -> Approved Symbiotic Arb: {m_id} | Side: {side} | EV: {best_ev:.2f} | Size: {final_size:.2f} (Allocated from ${bankroll})")

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
    bankroll = state.get("bankroll", DEFAULT_BANKROLL)
    
    try:
        poly = pmxt.Polymarket()
        
        # 1. VPIN Calculation with Sensitivity Tuning
        markets = state.get("current_markets", [])
        if markets:
            target_market = sorted(markets, key=lambda x: x.get('liquidity', 0), reverse=True)[0]
            token_id = target_market.get("yes_token_id")
            trades = poly.fetch_trades(token_id)
            if trades and len(trades) >= 10:
                # Handle objects from new PMXT
                buy_vol = sum(getattr(t, 'amount', 0) for t in trades if getattr(t, 'side', '') == 'buy')
                sell_vol = sum(getattr(t, 'amount', 0) for t in trades if getattr(t, 'side', '') == 'sell')
                total_vol = buy_vol + sell_vol
                vpin = abs(buy_vol - sell_vol) / total_vol if total_vol > 0 else 0
                
                # SENSITIVITY TUNING: 0.5 threshold instead of 0.7 during 'toxic' periods (final hour)
                is_proximate = False
                res_time_str = target_market.get("resolution_time")
                if res_time_str:
                    try:
                        # Normalize ISO string for fromisoformat (handle 'Z')
                        res_time_str = str(res_time_str).replace('Z', '+00:00')
                        if 'T' in res_time_str:
                            res_dt = datetime.fromisoformat(res_time_str)
                        else:
                            # Fallback for simple date strings
                            res_dt = datetime.strptime(res_time_str.split('+')[0].strip(), "%Y-%m-%d")
                        now = datetime.now(res_dt.tzinfo) if res_dt.tzinfo else datetime.now()
                        
                        time_to_close = (res_dt - now).total_seconds()
                        if 0 < time_to_close < 3600: # Within 60 minutes
                            is_proximate = True
                            print(f" -> TOXIC PERIOD DETECTED: {target_market['id']} resolves in {time_to_close/60:.1f} mins. Tuning VPIN sensitivity.")
                    except Exception as e:
                        print(f" -> [WARN] Could not parse resolution_time: {e}")
                
                vpin_threshold = 0.6 if is_proximate else 0.95 
                                
                if vpin > vpin_threshold:
                    risk_flags.append("VPIN_KILL_SWITCH")
                    send_telegram_msg(f"🚨 *KILL SWITCH*: VPIN {vpin:.2f} > {threshold}. Halting.")

        # 2. Exposure Monitoring
        total_exposure = sum(state.get("position_sizes", {}).values())
        if total_exposure > (bankroll * 0.35): # Slightly higher bucket allowed with compounding
            risk_flags.append("MAX_EXPOSURE_REACHED")
            send_telegram_msg(f"⚠️ *EXPOSURE ALERT*: Total exposure ${total_exposure:.2f} exceeds 35% limit.")

        # 3. Kill-Switch Activation
        if risk_flags:
            # Only PAUSE the system for VPIN kill switch (critical market toxicity)
            if "VPIN_KILL_SWITCH" in risk_flags:
                if not PAPER_TRADING_MODE and not SIMULATION_MODE:
                    try:
                        poly.cancel_all_orders()
                    except Exception as e:
                        print(f" -> [ERROR] Failed to cancel orders: {e}")
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
        poly = pmxt.Polymarket()
        
        for m_id, size in pos_sizes.items():
            is_exit = m_id.startswith("EXIT_")
            real_m_id = m_id.replace("EXIT_", "") if is_exit else m_id
            
            market = markets.get(real_m_id)
            if not market: continue
            
            side_type = trade_sides.get(m_id)
            
            # Refined Token and Side Selection
            is_yes = "yes" in str(side_type).lower()
            token_id = market["yes_token_id"] if is_yes else market["no_token_id"]
            order_side = "sell" if "sell" in str(side_type).lower() else "buy"
            
            # 1. Fetch Orderbook for Maker price (using specific token_id for precision)
            ob = poly.fetch_order_book(token_id)
            # Handle both object-style (new PMXT) and dict-style (old PMXT)
            bids = getattr(ob, 'bids', [])
            asks = getattr(ob, 'asks', [])
            
            # Fallback if book is empty
            if not bids or not asks:
                mkt_price = market.get("price_yes" if is_yes else "price_no", 0.5)
                limit_price = round(mkt_price - 0.01 if order_side == "buy" else mkt_price + 0.01, 3)
            else:
                # Maker Price Logic: Aggressive maker (1 tick better than best)
                # Ensure we don't cross the spread (post_only would fail anyway)
                best_bid = float(getattr(bids[0], 'price', 0))
                best_ask = float(getattr(asks[0], 'price', 0))
                
                if order_side == "buy":
                    # Place at best_bid + 0.001, but capped at best_ask - 0.001 to ensure maker status
                    limit_price = min(best_bid + 0.001, best_ask - 0.001)
                else:
                    # Place at best_ask - 0.001, but floor at best_bid + 0.001
                    limit_price = max(best_ask - 0.001, best_bid + 0.001)

            # 2. Boundary and Precision Check
            limit_price = round(max(0.005, min(0.995, limit_price)), 3)

            if PAPER_TRADING_MODE:
                # --- REALISTIC PAPER SIMULATION ---
                # 1. Missed Order Simulation (5% chance)
                if np.random.random() < 0.05:
                    print(f" -> [PAPER] Order MISSED for {real_m_id} (Network/Latency simulation)")
                    continue

                # 2. Slippage Simulation
                available_liq = market.get("ask_liquidity" if order_side == "buy" else "bid_liquidity", 1.0)
                slippage_factor = (size / max(1.0, available_liq)) * 0.02 
                
                # 3. Partial Fill Simulation
                fill_pct = np.random.uniform(0.6, 1.0) # Fill between 60% and 100%
                actual_filled_size = size * fill_pct

                execution_price = limit_price * (1 + slippage_factor if order_side == "buy" else 1 - slippage_factor)
                execution_price = round(max(0.005, min(0.995, execution_price)), 3)

                # 4. Latency Simulation
                latency = np.random.uniform(0.5, 2.0) 
                time.sleep(latency)
                
                print(f" -> [PAPER] Maker Order: {real_m_id} | {order_side} | Filled: {actual_filled_size:.2f}/{size:.2f} | Exec: {execution_price} | Slippage: {slippage_factor*100:.2f}%")
                
                # Update Virtual Wallet
                current_bal = db.get_latest_bankroll(mode='PAPER', default=DEFAULT_BANKROLL)
                if order_side == "buy":
                    new_bal = current_bal - actual_filled_size
                else: 
                    new_bal = current_bal + (actual_filled_size * execution_price) 
                
                db.update_bankroll(new_bal, new_bal, mode='PAPER')
                db.log_trade(real_m_id, side_type, actual_filled_size, execution_price, "paper_executed", mode='PAPER')
                executed_trades.append({"market": real_m_id, "side": order_side, "size": actual_filled_size, "price": execution_price, "status": "paper"})

            elif SIMULATION_MODE:
                print(f" -> [SIMULATION] Maker Order: {real_m_id} ({'YES' if is_yes else 'NO'}) | {order_side} | Price: {limit_price}")
                executed_trades.append({"market": real_m_id, "side": order_side, "size": size, "price": limit_price, "status": "simulation"})
            else:
                try:
                    order = poly.create_order(
                        token_id=token_id,
                        side=order_side,
                        size=size,
                        price=limit_price,
                        post_only=True # ENSURE MAKER ONLY
                    )
                    status = order.get('status', 'unknown')
                    executed_trades.append({"market": real_m_id, "side": order_side, "size": size, "price": limit_price, "status": status})
                    db.log_trade(real_m_id, order_side, size, limit_price, status)
                except Exception as e:
                    print(f" -> [FAIL] Maker order failed for {real_m_id}: {e}")

        # Notify via Telegram
        if executed_trades:
            prefix = "[PAPER] " if PAPER_TRADING_MODE else ""
            msg = f"✅ {prefix}*TRADES EXECUTED (v2.0 Maker-Only)*\n"
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
    # Only block execution for critical kill switches
    critical_flags = ["VPIN_KILL_SWITCH", "GUARDIAN_FETCH_ERROR"]
    active_criticals = [f for f in state.get("risk_flags", []) if f in critical_flags]
    
    if active_criticals:
        print(f"!!! HALT: Execution skipped due to Critical Risk Flags: {active_criticals} !!!")
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
        "total_exposure": 0.0, "api_weights": {}, "open_positions": [], "bankroll": DEFAULT_BANKROLL
    }
    
    try:
        final_state = app.invoke(initial_state)
        for log in final_state["cycle_logs"]:
            print(f" - {log}")
        
        # REMOVED: Redundant bankroll update that was overwriting DB with stale state
        # The Executor node now handles all real-time bankroll updates in the DB.
        return final_state.get("pause_flag", False)

    except Exception as e:
        print(f"CRITICAL ERROR in Cycle: {e}")
        return True 

if __name__ == "__main__":
    print("--------------------------------------------------")
    print("Weather Arbitrage Autonomous AI Agent v2.0 (High Velocity)")
    current_mode = 'LIVE'
    if SIMULATION_MODE: current_mode = 'SIMULATION'
    if PAPER_TRADING_MODE: current_mode = 'PAPER TRADING'
    print(f"Mode: {current_mode}")
    print("--------------------------------------------------")
    
    POLL_INTERVAL = 60 
    system_paused = False
    
    # Schedule Reliability Backtest Loop every 6 hours
    schedule.every(6).hours.do(ReliabilityManager.run_backtest_loop)
    
    # Run once at startup
    print("[INIT] Starting Reliability Backtest...")
    ReliabilityManager.run_backtest_loop()
    print("[INIT] Reliability Backtest complete.")

    print("[INIT] Starting main loop...")
    while True:
        try:
            # Run scheduled tasks
            print(f"[LOOP] Checking scheduled tasks at {time.strftime('%H:%M:%S')}...")
            schedule.run_pending()
            
            print("[LOOP] Invoking run_agent_loop...")
            system_paused = run_agent_loop(system_paused)
            print(f"[LOOP] run_agent_loop finished. System paused: {system_paused}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Loop error: {e}")
            system_paused = True 
        
        time.sleep(POLL_INTERVAL)
