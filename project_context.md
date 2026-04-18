# Autonomous AI Trading Agent Context

## Project Overview
Building a fully autonomous AI trading agent that operates 24/7 on Polymarket to execute a weather arbitrage strategy, using advanced quantitative mathematics to generate consistent edge on temperature and precipitation markets.

## System Architecture
- **Core Engine**: Multi-agent LangGraph system.
- **Agent Roles**: 
    - **Researcher**: Fetches weather data.
    - **Analyst**: Models distributions.
    - **Decision**: Determines trade signals.
    - **Executor**: Places orders.
    - **RiskGuardian**: Monitors safety and caps.
- **Integration**: PMXT SDK for Polymarket data and order execution on Polygon.
- **Networking**: Proton VPN Free for geo-restriction bypass.

## Quantitative Strategy
- **Cycle**: Executes every 15 minutes.
- **Data Ingestion**: Forecasts from multiple free weather APIs.
- **Probability Modeling**: 
    - Gaussian ensemble modeling for temperature distributions.
    - Bayesian updating for refined probability estimation.
- **Trade Execution**:
    - Expected-value (EV) threshold filtering.
    - Fractional Kelly position sizing.
    - Limit orders only (when positive edge is confirmed).
- **Capital**: $50 USDC starting capital (capped).

## Safety and Risk Management
- **Kill Switch**: VPIN-based (Volume Probability of Informed Trading) automatic shutdown.
- **Alerts**: Human-in-the-loop notification system via Telegram.
- **Guardian**: Continuous risk monitoring by the RiskGuardian agent.
