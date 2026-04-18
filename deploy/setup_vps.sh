#!/bin/bash
# Setup script for a clean Ubuntu/Debian VPS

# 1. System Updates
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y python3-pip python3-venv git sqlite3 tailscale # tailscale for VPN

# 2. Setup Tailscale (VPN)
# Replace with your auth key or run 'sudo tailscale up' manually
echo "Please run 'sudo tailscale up' manually to authenticate the VPN."

# 3. Install Node.js & PM2 (Process Manager)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
sudo npm install -g pm2

# 4. Clone and Install App
git clone https://github.com/kwstx/tradingbot.git
cd tradingbot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 5. Start with PM2
pm2 start deploy/ecosystem.config.js
pm2 save
pm2 startup
