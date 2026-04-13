#!/usr/bin/env bash
#
# Full daily pipeline: update data → patch scores → rebuild dataset →
# export predictions to JSON → git push → triggers Vercel deploy.
#
# Scheduled by launchd (06:00 daily) or run manually.

set -euo pipefail

PROJECT_DIR="/Users/lchao/Documents/projects/NBA-Machine-Learning-Sports-Betting"
LOG_DIR="$HOME/Library/Logs/nba-ml"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/pipeline-$(date +%Y-%m).log"
STAMP=$(date '+%Y-%m-%d %H:%M:%S')
echo "===== $STAMP daily_pipeline start =====" >> "$LOG_FILE"

notify_fail() {
    local step="$1"
    /usr/bin/osascript -e "display notification \"Step failed: $step\" with title \"NBA-ML Pipeline\" sound name \"Basso\"" || true
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

# 1. Update raw data (regular season + playoff stats snapshots).
cd src/Process-Data
run_step "Get_Data"           python -m Get_Data
run_step "Get_Odds_Data"      python -m Get_Odds_Data
run_step "Get_Playoffs_Data"  python -m Get_Playoffs_Data || true
cd "$PROJECT_DIR"

# 2. Patch final scores for recent days (sbrscrape).
run_step "Patch_Scores"  python scripts/patch_scores.py

# 3. Derive playoff series state from updated odds (idempotent).
run_step "Get_Series_State" python src/Process-Data/Get_Series_State.py --all-seasons || true

# 4. Rebuild the merged dataset (now includes is_playoff + series cols).
run_step "Create_Games"  python src/Process-Data/Create_Games.py

# 5. Export predictions to static JSON for Vercel.
run_step "Export_JSON"   python scripts/export_predictions.py

# 6. Git commit & push web/data/ to trigger Vercel deploy.
cd "$PROJECT_DIR"
if git diff --quiet web/data/ 2>/dev/null; then
    echo "No changes in web/data/, skipping push." >> "$LOG_FILE"
else
    git add web/data/
    git commit -m "daily update $(date +%Y-%m-%d)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>" >> "$LOG_FILE" 2>&1
    run_step "Git_Push" git push
fi

echo "===== $(date '+%Y-%m-%d %H:%M:%S') daily_pipeline OK =====" >> "$LOG_FILE"
