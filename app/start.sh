#!/usr/bin/env bash
# FormAI Backend Launcher
# Binds to 0.0.0.0 so any device on the same Wi-Fi can reach it.

set -e

cd "$(dirname "$0")"

# ── Detect LAN IP ─────────────────────────────────────────────────────────────
LAN_IP=$(python3 -c "
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(('8.8.8.8', 80))
print(s.getsockname()[0])
s.close()
" 2>/dev/null || echo "127.0.0.1")

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║            FormAI Backend  v2.0                 ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║  Local  : http://localhost:8000                  ║"
echo "║  Mobile : http://$LAN_IP:8000          ║"
echo "║                                                  ║"
echo "║  Set Flutter baseUrl to:                         ║"
echo "║  http://$LAN_IP:8000                   ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── Launch ────────────────────────────────────────────────────────────────────
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
