#!/bin/bash
# Ensures VPN is up before running the agent

check_vpn() {
    # Check if tailscale is connected
    if tailscale status | grep -q "Tailscale is stopped"; then
        return 1
    fi
    return 0
}

echo "Checking VPN status..."
if ! check_vpn; then
    echo "VPN is down. Reconnecting..."
    sudo tailscale up --authkey=$TAILSCALE_AUTHKEY
fi

# Give it a moment to stabilize
sleep 5

# Start the agent (usually called via PM2)
exec "$@"
