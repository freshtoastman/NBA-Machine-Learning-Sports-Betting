#!/usr/bin/env bash
#
# NBA-ML daily data update.
# Runs Get_Data, Get_Odds_Data, Create_Games in sequence.
# Notifies via macOS Notification Center on failure.
# Scheduled by ~/Library/LaunchAgents/com.lchao.nba-ml.daily.plist (06:00 daily).

set -euo pipefail

PROJECT_DIR="/Users/lchao/Documents/projects/NBA-Machine-Learning-Sports-Betting"
LOG_DIR="$HOME/Library/Logs/nba-ml"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/daily-$(date +%Y-%m).log"
echo "===== $(date '+%Y-%m-%d %H:%M:%S') daily_update start =====" >> "$LOG_FILE"

notify_fail() {
    local step="$1"
    /usr/bin/osascript -e "display notification \"Step failed: $step. See $LOG_FILE\" with title \"NBA-ML Daily Update\" sound name \"Basso\"" || true
}

cd "$PROJECT_DIR"
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH="$PROJECT_DIR"

run_step() {
    local name="$1"; shift
    echo "--- $name ---" >> "$LOG_FILE"
    if ! "$@" >> "$LOG_FILE" 2>&1; then
        notify_fail "$name"
        echo "===== $(date '+%Y-%m-%d %H:%M:%S') FAILED at $name =====" >> "$LOG_FILE"
        exit 1
    fi
}

# Get_Data and Get_Odds_Data work from src/Process-Data; Create_Games imports
# `from src.Utils...` so it must run from project root (with PYTHONPATH set).
cd src/Process-Data
run_step "Get_Data"      python -m Get_Data
run_step "Get_Odds_Data" python -m Get_Odds_Data
cd "$PROJECT_DIR"
run_step "Create_Games"  python src/Process-Data/Create_Games.py

echo "===== $(date '+%Y-%m-%d %H:%M:%S') daily_update OK =====" >> "$LOG_FILE"
