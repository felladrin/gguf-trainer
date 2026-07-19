#!/usr/bin/env bash
# Monitor Phase A pretraining: log progress, detect anomalies
# Run via cron: */30 * * * * /home/victor/Repositories/gguf-trainer/scripts/monitor_phase_a.sh

set -uo pipefail

LOG="$HOME/Repositories/gguf-trainer/examples/phaseA.log"
OUTPUT="$HOME/Repositories/gguf-trainer/examples/phaseA.monitor.log"
ALERT="$HOME/Repositories/gguf-trainer/examples/phaseA.alert"

# Check if training is running
if ! pgrep -f 'pretrain[.]ts' >/dev/null; then
  echo "$(date '+%Y-%m-%d %H:%M:%S') [ALERT] Training process not running!" >> "$ALERT"
  exit 1
fi

# Get GPU utilization
GPU=$(cat /sys/class/drm/card*/device/gpu_busy_percent 2>/dev/null | head -1)
if [ -z "$GPU" ]; then
  echo "$(date '+%Y-%m-%d %H:%M:%S') [WARN] Could not read GPU utilization" >> "$OUTPUT"
else
  if [ "$GPU" -lt 80 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ALERT] GPU utilization low: ${GPU}%" >> "$ALERT"
  fi
fi

# Get latest step and loss
LATEST=$(tail -1 "$LOG" 2>/dev/null)
STEP=$(echo "$LATEST" | grep -oE 'step\s+[0-9]+' | grep -oE '[0-9]+' || echo "N/A")
LOSS=$(echo "$LATEST" | grep -oE 'loss\s+[0-9.]+' | grep -oE '[0-9.]+' || echo "N/A")

# Check for NaN
if echo "$LATEST" | grep -qi 'nan'; then
  echo "$(date '+%Y-%m-%d %H:%M:%S') [ALERT] NaN detected in loss!" >> "$ALERT"
fi

# Log progress
echo "$(date '+%Y-%m-%d %H:%M:%S') step=$STEP loss=$LOSS gpu=${GPU}% " >> "$OUTPUT"

# Check if we've completed
TOTAL=88000
if [ "$STEP" != "N/A" ] && [ "$STEP" -ge "$TOTAL" ]; then
  echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Training completed at step $STEP" >> "$OUTPUT"
  echo "$(date '+%Y-%m-%d %H:%M:%S') [ALERT] Training completed!" >> "$ALERT"
fi

echo "OK: step=$STEP loss=$LOSS gpu=${GPU}%"
