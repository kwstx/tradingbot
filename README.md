# tradingbot

## Autonomous AI Trading Agent for Polymarket Weather Arbitrage

This repository contains a multi-agent system (Researcher, Analyst, Decision, Executor, RiskGuardian) designed to execute weather arbitrage strategies on Polymarket.

### Features
- 24/7 autonomous monitoring and execution.
- Gaussian ensemble modeling and Bayesian updating for weather probabilities.
- Risk management via VPIN-based kill switch and fractional Kelly sizing.
- Real-time market interaction via PMXT SDK on Polygon.

### Setup
1. Create a virtual environment: `python -m venv polymarket-weather-agent`.
2. Activate it: `.\polymarket-weather-agent\Scripts\activate`.
3. Install dependencies: `pip install -r requirements.txt`.
4. Configure your `.env` file with the necessary API keys and private key.
