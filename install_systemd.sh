#!/bin/bash
# install_systemd.sh -- installs the Celery worker, Discord bot, and file
# server as systemd services, so they start on boot and auto-restart if they
# crash. Run this once (with sudo) from inside the project folder:
#
#   cd ~/deepseekyt
#   sudo ./install_systemd.sh
#
# After this, you never need to run start_worker.sh / start_bot.sh /
# file_server.py by hand again -- systemd manages all three.
set -euo pipefail

if [ "$EUID" -ne 0 ]; then
    echo "Please run this with sudo: sudo ./install_systemd.sh"
    exit 1
fi

# The directory this script lives in (i.e. the project root).
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The user who invoked sudo (so services don't run as root).
RUN_USER="${SUDO_USER:-$(whoami)}"
VENV_PY="$PROJECT_DIR/venv/bin/python"
VENV_CELERY="$PROJECT_DIR/venv/bin/celery"
ENV_FILE="$PROJECT_DIR/.env"

if [ ! -x "$VENV_PY" ]; then
    echo "❌ $VENV_PY not found. Run ./setup.sh first to create venv/."
    exit 1
fi
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ $ENV_FILE not found. Copy .env.example to .env and fill it in first."
    exit 1
fi

echo "📦 Installing systemd services for user '$RUN_USER' in $PROJECT_DIR"

# --- Celery worker ---------------------------------------------------------
cat > /etc/systemd/system/shorts-worker.service <<EOF
[Unit]
Description=YouTube Shorts Bot - Celery worker
After=network.target redis-server.service
Wants=redis-server.service

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$PROJECT_DIR
EnvironmentFile=$ENV_FILE
ExecStart=$VENV_CELERY -A tasks worker --loglevel=info
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# --- Discord bot -------------------------------------------------------------
cat > /etc/systemd/system/shorts-bot.service <<EOF
[Unit]
Description=YouTube Shorts Bot - Discord bot
After=network.target shorts-worker.service
Wants=shorts-worker.service

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$PROJECT_DIR
EnvironmentFile=$ENV_FILE
ExecStart=$VENV_PY discord_bot.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# --- File share server -------------------------------------------------------
cat > /etc/systemd/system/shorts-fileserver.service <<EOF
[Unit]
Description=YouTube Shorts Bot - temp file share server
After=network.target

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$PROJECT_DIR
EnvironmentFile=$ENV_FILE
ExecStart=$VENV_PY file_server.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

echo "🔄 Reloading systemd..."
systemctl daemon-reload

echo "✅ Enabling services to start on boot..."
systemctl enable redis-server shorts-fileserver shorts-worker shorts-bot

echo "🚀 Starting services now..."
systemctl restart redis-server
systemctl restart shorts-fileserver
systemctl restart shorts-worker
systemctl restart shorts-bot

echo ""
echo "Done! All three services are running and will survive reboots/crashes."
echo ""
echo "Useful commands:"
echo "  systemctl status shorts-bot shorts-worker shorts-fileserver"
echo "  journalctl -u shorts-bot -f          # live bot logs"
echo "  journalctl -u shorts-worker -f       # live worker logs"
echo "  journalctl -u shorts-fileserver -f   # live file-server logs"
echo "  sudo systemctl restart shorts-bot shorts-worker shorts-fileserver"
