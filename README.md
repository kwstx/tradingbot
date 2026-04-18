# tradingbot

## Autonomous AI Trading Agent for Polymarket Weather Arbitrage

This repository contains a multi-agent system (Researcher, Analyst, Decision, Executor, RiskGuardian) designed to execute weather arbitrage strategies on Polymarket.

### Features
- 24/7 autonomous monitoring and execution via LangGraph.
- Gaussian ensemble modeling and Bayesian updating for weather probabilities.
- Risk management via VPIN-based kill switch and fractional Kelly sizing.
- Real-time market interaction via PMXT SDK on Polygon.
- **Persistence**: SQLite-backed tracking of trades, bankroll history, and Bayesian priors.
- **Backtesting**: Replay historical data to calculate Sharpe Ratio and ROI.
- **Simulation Mode**: Risk-free monitoring with actual order calls disabled.
- **Daily Reporting**: Automated Telegram summaries of bot performance.

### Setup
1. Create a virtual environment: `python -m venv venv`.
2. Activate it: `source venv/bin/activate` (Linux) or `.\venv\Scripts\activate` (Windows).
3. Install dependencies: `pip install -r requirements.txt`.
4. Configure your `.env` file:
   - `SIMULATION_MODE=true` (Default)
   - `TELEGRAM_BOT_TOKEN=your_token`
   - `TELEGRAM_CHAT_ID=your_id`
   - `PMXT_PRIVATE_KEY=your_key`
5. Initialize the database: Persistence is handled automatically on first run.

### Backtesting
Run the backtest script with historical market and weather data:
```bash
python backtest.py
```

### Deployment
For 24/7 operation on a VPS:
1. Use `deploy/setup_vps.sh` for environment preparation.
2. Manage the process with PM2:
   ```bash
   pm2 start deploy/ecosystem.config.js
   ```
3. Monitor logs:
   ```bash
   pm2 logs
   ```
