module.exports = {
  apps : [{
    name: "weather-arb-agent",
    script: "./weather_arb_agent.py",
    interpreter: "./venv/bin/python3",
    env: {
      NODE_ENV: "production",
      SIMULATION_MODE: "true",
      HUMAN_APPROVAL_REQUIRED: "true"
    },
    env_live: {
      SIMULATION_MODE: "false",
      HUMAN_APPROVAL_REQUIRED: "true"
    },
    restart_delay: 10000,
    max_restarts: 10,
    cron_restart: "0 0 * * *", // Daily restart to refresh sessions/cleanup
    log_date_format: "YYYY-MM-DD HH:mm:ss",
    merge_logs: true,
  }]
};
