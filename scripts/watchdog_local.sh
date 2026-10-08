#!/usr/bin/env bash
# Keeps the two local Study B runs going on the laptop (15.7 GB WSL memory budget).
# Every 5 minutes: log free memory and progress; restart a run that stopped before
# finishing (both resume where they left off), at most MAX_RESTARTS times each.
# Start:  nohup bash scripts/watchdog_local.sh > /dev/null 2>&1 &
# Log:    outputs/watchdog.log

cd "$(dirname "$0")/.." || exit 1
LOG=outputs/watchdog.log
MAX_RESTARTS=5
STUDYB_TOTAL=6720
TABPFN_TOTAL=4032
restarts_b=0
restarts_t=0

progress() {  # prints: studyB_ok_units tabpfn_rows
  .venv/bin/python - <<'EOF'
import glob, pandas as pd
u = pd.concat([pd.read_csv(f) for f in glob.glob("results/raw/studyB/*/units.csv") + glob.glob("outputs/phase5/units.csv")])
u = u[u.ok].drop_duplicates(["did", "rep", "dtype", "level"])
try:
    t = len(pd.read_csv("outputs/phase5/tabpfn_nofp.csv").drop_duplicates(["did", "rep", "dtype", "level"]))
except FileNotFoundError:
    t = 0
print(len(u), t)
EOF
}

running() { ps -eo args | grep -v grep | grep -q "$1"; }

while true; do
  read -r done_b done_t < <(progress)
  avail=$(free -m | awk 'NR==2{print $7}')
  echo "$(date '+%F %T') available_MB=$avail studyB=$done_b/$STUDYB_TOTAL tabpfn=$done_t/$TABPFN_TOTAL" >> "$LOG"

  if [ "$done_b" -lt "$STUDYB_TOTAL" ] && ! running "scripts/phase5_studyB.py" && [ $restarts_b -lt $MAX_RESTARTS ]; then
    restarts_b=$((restarts_b + 1))
    echo "$(date '+%F %T') restarting Study B (restart $restarts_b)" >> "$LOG"
    PYTHONPATH=src nohup .venv/bin/python scripts/phase5_studyB.py --reps 0,2 --workers 2 --timeout-min 40 \
      >> outputs/phase5_local.log 2>&1 &
  fi
  if [ "$done_t" -lt "$TABPFN_TOTAL" ] && ! running "scripts/phase5_tabpfn.py" && [ $restarts_t -lt $MAX_RESTARTS ]; then
    restarts_t=$((restarts_t + 1))
    echo "$(date '+%F %T') restarting TabPFN (restart $restarts_t)" >> "$LOG"
    PYTHONPATH=src nohup .venv-gpu/bin/python scripts/phase5_tabpfn.py --reps 0,1,2 >> outputs/phase5_tabpfn_nofp.log 2>&1 &
  fi

  if [ "$done_b" -ge "$STUDYB_TOTAL" ] && [ "$done_t" -ge "$TABPFN_TOTAL" ]; then
    echo "$(date '+%F %T') both runs complete; watchdog exits" >> "$LOG"
    exit 0
  fi
  sleep 300
done
