#!/usr/bin/env bash
set -euo pipefail

cleanup() {
  if [[ -n "${P1:-}" ]]; then kill "$P1" 2>/dev/null || true; fi
  if [[ -n "${P2:-}" ]]; then kill "$P2" 2>/dev/null || true; fi
  if [[ -n "${P3:-}" ]]; then kill "$P3" 2>/dev/null || true; fi
}
trap cleanup EXIT

mkdir -p runs

python3 examples/udp_node.py \
  --node-id ECG \
  --bind-port 6001 \
  --peers "127.0.0.1:6003,127.0.0.1:6005" \
  --seed 100 > runs/udp_ecg.log 2>&1 &
P1=$!

python3 examples/udp_node.py \
  --node-id Motion \
  --bind-port 6003 \
  --peers "127.0.0.1:6001,127.0.0.1:6005" \
  --seed 200 > runs/udp_motion.log 2>&1 &
P2=$!

python3 examples/udp_node.py \
  --node-id Breath \
  --bind-port 6005 \
  --peers "127.0.0.1:6001,127.0.0.1:6003" \
  --seed 300 > runs/udp_breath.log 2>&1 &
P3=$!

wait "$P1" "$P2" "$P3"

echo "UDP demo finished. Logs:"
echo "  runs/udp_ecg.log"
echo "  runs/udp_motion.log"
echo "  runs/udp_breath.log"
