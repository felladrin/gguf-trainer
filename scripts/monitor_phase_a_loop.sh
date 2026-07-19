#!/usr/bin/env bash
# Continuous monitor for Phase A pretraining
# Run: ./scripts/monitor_phase_a_loop.sh
# Stops when training stops or Ctrl+C

set -uo pipefail

# Repo root = parent of this script's dir, so it works from any clone location.
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$REPO/examples/phaseA.log"
MONITOR_LOG="$REPO/examples/phaseA.monitor.log"
ALERT_LOG="$REPO/examples/phaseA.alert"

echo "=== Phase A Monitor Started at $(date) ===" | tee -a "$MONITOR_LOG"
echo "Watching: $LOG" | tee -a "$MONITOR_LOG"

LAST_STEP=0
LAST_TIME=$(date +%s)

while true; do
  # Check if training is running
  if ! pgrep -f 'pretrain[.]ts' >/dev/null; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ALERT] Training process stopped!" | tee -a "$ALERT_LOG"
    echo "Training is no longer running. Check $ALERT_LOG"
    exit 1
  fi

  # Get GPU utilization
  GPU=$(cat /sys/class/drm/card*/device/gpu_busy_percent 2>/dev/null | head -1)
  
  # Get latest line with step info
  LATEST=$(tail -1 "$LOG" 2>/dev/null)
  STEP=$(echo "$LATEST" | grep -oE 'step\s+[0-9]+' | grep -oE '[0-9]+' || echo "N/A")
  LOSS=$(echo "$LATEST" | grep -oE 'loss\s+[0-9.]+' | grep -oE '[0-9.]+' || echo "N/A")

  # Check for NaN
  if echo "$LATEST" | grep -qi 'nan'; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ALERT] NaN detected in loss!" | tee -a "$ALERT_LOG"
  fi

  # Check for stalled training (same step for > 15 minutes)
  if [ "$STEP" != "N/A" ] && [ "$STEP" -gt 0 ]; then
    if [ "$STEP" -eq "$LAST_STEP" ]; then
      NOW=$(date +%s)
      DIFF=$(( (NOW - LAST_TIME) / 60 ))
      if [ "$DIFF" -gt 15 ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [ALERT] Training stalled at step $STEP for ${DIFF} minutes!" | tee -a "$ALERT_LOG"
      fi
    else
      LAST_STEP=$STEP
      LAST_TIME=$(date +%s)
    fi
  fi

  # Log progress every minute
  echo "$(date '+%Y-%m-%d %H:%M:%S') step=$STEP loss=$LOSS gpu=${GPU}%" >> "$MONITOR_LOG"

  # Print status every 5 minutes (every 5th iteration)
  if [ $(( $(date +%s) % 300 )) -lt 10 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') ✓ step=$STEP loss=$LOSS gpu=${GPU}%"
  fi

  sleep 60
done
