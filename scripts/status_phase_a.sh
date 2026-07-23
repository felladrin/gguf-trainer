#!/usr/bin/env bash
# Quick status check for Phase A training
# Run: ./scripts/status_phase_a.sh

# Repo root = parent of this script's dir, so it works from any clone location.
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$REPO/examples/phaseA.log"
CKPT="$REPO/examples/pretrain-blend-base.gguf"

echo "=== Phase A Training Status ==="
echo "Time: $(date)"
echo ""

# Check if running
if pgrep -f 'pretrain[.]ts' >/dev/null; then
  PIDS=$(pgrep -f 'pretrain[.]ts' | tr '\n' ' ')
  echo "Status: RUNNING (PIDs: $PIDS)"
else
  echo "Status: NOT RUNNING"
fi

# GPU utilization
GPU=$(cat /sys/class/drm/card*/device/gpu_busy_percent 2>/dev/null | head -1)
echo "GPU: ${GPU:-N/A}%"

# Latest progress
LATEST=$(tail -1 "$LOG" 2>/dev/null)
STEP=$(echo "$LATEST" | grep -oE 'step\s+[0-9]+' | grep -oE '[0-9]+' || echo "N/A")
LOSS=$(echo "$LATEST" | grep -oE 'loss\s+[0-9.]+' | grep -oE '[0-9.]+' || echo "N/A")
echo "Step: $STEP / 88000 ($(echo "scale=1; $STEP * 100 / 88000" | bc)%)"
echo "Loss: $LOSS"

# Checkpoint info
if [ -f "$CKPT" ]; then
  SIZE=$(ls -lh "$CKPT" | awk '{print $5}')
  MOD=$(ls -l "$CKPT" | awk '{print $6, $7, $8}')
  echo "Checkpoint: $CKPT ($SIZE, last updated: $MOD)"
else
  echo "Checkpoint: NOT FOUND"
fi

# Recent alerts
if [ -f "$REPO/examples/phaseA.alert" ]; then
  ALERTS=$(wc -l < "$REPO/examples/phaseA.alert")
  echo "Alerts: $ALERTS (see phaseA.alert)"
fi

echo ""
echo "Commands:"
echo "  Monitor: tail -f $LOG"
echo "  Stop:    bash $REPO/stop_phase_a.sh"
echo "  Resume:  bash $REPO/resume_phase_a.sh"
