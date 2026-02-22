#!/usr/bin/env bash
# Restart the Veo Soccer Analysis UI server.
# Kills any running instance, then starts a fresh one.

set -euo pipefail
cd "$(dirname "$0")/.."

# Kill existing run_ui.py processes
pkill -f 'python.*run_ui\.py' 2>/dev/null && echo "Stopped running server." || echo "No running server found."

# Brief pause to release the port
sleep 1

# Start the server in foreground (Ctrl-C to stop)
echo "Starting UI server..."
PYTHONPATH=".venv/lib/python3.11/site-packages" /opt/homebrew/bin/python3.11 run_ui.py
