#!/usr/bin/env bash
#
# Full daily pipeline: update data → patch scores → rebuild dataset →
# export predictions to JSON → git push → triggers Vercel deploy.
#
# Scheduled by launchd (06:00 daily) or run manually.
#
# Usage: daily_pipeline.sh [morning|afternoon]
#   morning   (default, 06:00 Taipei = 18:00 ET the previous day): fetch today's
#             stats table (holds games through two US days ago), scores, odds.
#   afternoon (15:00 Taipei = 03:00 ET, all of yesterday's games final): re-fetch
#             today's stats table so pre-game features include every game through
#             yesterday, refresh lines, re-export. Without it, games tipping before
#             18:00 ET (128 of 1,207 in 2026-27) freeze with a one-day-stale table
#             and 30-hour-old lines.

set -euo pipefail

MODE="${1:-morning}"
case "$MODE" in
    morning|afternoon) ;;
    *) echo "unknown mode: $MODE (morning|afternoon)" >&2; exit 2 ;;
esac

# The 2026-10-19 06:00 run took 2 h 54 min for ~2 min of work: the laptop was asleep on
# battery, launchd started the job inside a 12 s dark wake, and the process was frozen
# between dark wakes (collect_referee_data 06:06 -> 08:19, Create_Games 2426 s vs 67 s).
# Re-exec under caffeinate so the run holds a no-idle-sleep assertion from its first
# step to the final OK line (-i idle, -m disk). It cannot wake a sleeping Mac (that
# needs mains power or a pmset wake schedule), only keep an awake one awake.
# 2026-10-25: ON AC the 06:00 run still took 22 min for ~2 min of work. pmset -g log:
# 06:00:21 DarkWake (AC, 45 s) -> 06:01:06 'Maintenance Sleep' -> 06:09:24 DarkWake
# (45 s) -> 06:10:09 Sleep -> 06:21:32 DarkWake -> 06:22:12 OK. A dark wake ends with a
# *maintenance* sleep that -i (idle sleep only) does not block; -s (PreventSystemSleep)
# does, and it is valid exactly when on AC. On battery -s is ignored (man caffeinate),
# so it costs nothing there and the power line below still explains those mornings.
if [ -z "${NBA_ML_CAFFEINATED:-}" ] && [ -x /usr/bin/caffeinate ] && [ -f "$0" ]; then
    export NBA_ML_CAFFEINATED=1
    exec /usr/bin/caffeinate -i -m -s /bin/bash "$0" "$@"
fi
# bash 3.2 (macOS /bin/bash) + `set -u`: expanding an EMPTY array with "${ARR[@]}"
# is a fatal "unbound variable" (fixed only in bash 4.4). Keep the flag as a plain
# string and expand it unquoted (it holds no spaces), so morning mode passes nothing.
REFRESH_FLAG=""
if [ "$MODE" = "afternoon" ]; then
    REFRESH_FLAG="--refresh-latest"
fi

PROJECT_DIR="/Users/lchao/Documents/projects/NBA-Machine-Learning-Sports-Betting"
LOG_DIR="$HOME/Library/Logs/nba-ml"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/pipeline-$(date +%Y-%m).log"
STAMP=$(date '+%Y-%m-%d %H:%M:%S')
echo "===== $STAMP daily_pipeline start ($MODE) =====" >> "$LOG_FILE"
# On battery a closed MacBook only gets 2 s dark wakes every ~16 min (pmset -g log,
# 2026-10-24: 06:13:30 DarkWake -> 06:13:32 Sleep -> 06:29:36 DarkWake ...), so every
# step below stalls for a multiple of ~960 s. caffeinate -s is AC-only; nothing in
# this script can fix it. Record the power source so the log explains a 3 h run.
echo "--- power: $(pmset -g batt 2>/dev/null | head -1 | sed "s/Now drawing from //") ---" >> "$LOG_FILE"

notify_fail() {
    local step="$1"
    /usr/bin/osascript -e "display notification \"Step failed: $step\" with title \"NBA-ML Pipeline\" sound name \"Basso\"" || true
}

# Any exit that is not the final OK line (set -e/-u aborts outside run_step, a kill
# from a sleeping laptop, a missing .venv) gets a FAILED marker in the log and a
# notification; otherwise the log just ends after "start" and nobody notices.
PIPELINE_OK=0
on_exit() {
    local rc=$?
    if [ "$PIPELINE_OK" -ne 1 ]; then
        # bash 3.2 reports rc=0 inside the trap after a set -u abort and would then
        # exit 0 (launchd sees success); force a failure status.
        [ "$rc" -eq 0 ] && rc=1
        echo "===== $(date '+%Y-%m-%d %H:%M:%S') FAILED (exit $rc, unexpected abort) =====" >> "$LOG_FILE"
        notify_fail "unexpected abort (exit $rc)"
        exit "$rc"
    fi
}
trap on_exit EXIT

cd "$PROJECT_DIR"
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH="$PROJECT_DIR"

# Each step logs its start time and elapsed seconds: the 2026-10-17 06:00 run took
# 535 s against ~150 s on the two days before and the log could not say which step
# was slow (likely a sluggish stats.nba.com request; see the 30 s timeout in tools.py).
run_step() {
    local name="$1"; shift
    local t0
    t0=$(date +%s)
    echo "--- $name ($(date '+%H:%M:%S')) ---" >> "$LOG_FILE"
    if ! "$@" >> "$LOG_FILE" 2>&1; then
        notify_fail "$name"
        echo "===== $(date '+%Y-%m-%d %H:%M:%S') FAILED at $name after $(( $(date +%s) - t0 )) s =====" >> "$LOG_FILE"
        PIPELINE_OK=1   # already logged + notified; keep the EXIT trap quiet
        exit 1
    fi
    echo "--- $name done in $(( $(date +%s) - t0 )) s ---" >> "$LOG_FILE"
}

# 0. Wait for the network. The 2026-10-23 06:00 run had no DNS for its whole 3 h 17 min
# (every host "nodename nor servname provided"): Get_Advanced_Data and Get_Odds_Data
# burned 28 min in retries, Create_Games rebuilt from stale tables, Git_Push failed and
# the first model slate shipped nothing. Wi-Fi after a wake can take minutes to
# re-associate; probe a host the pipeline needs and wait up to NBA_ML_NET_WAIT_MAX
# seconds (default 10 min) before the first fetch. Never fatal: a run with no network
# still writes its FAILED marker where the steps fail, and the probe is skipped
# when the budget is 0 (tests).
NET_WAIT_MAX="${NBA_ML_NET_WAIT_MAX:-600}"
net_up() {
    /usr/bin/curl -sS -m 8 -o /dev/null -A "Mozilla/5.0" \
        "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json" >/dev/null 2>&1
}
if [ "$NET_WAIT_MAX" -gt 0 ]; then
    t_net=$(date +%s)
    if ! net_up; then
        echo "--- network not reachable at $(date '+%H:%M:%S'); waiting up to ${NET_WAIT_MAX}s ---" >> "$LOG_FILE"
        while ! net_up; do
            if [ $(( $(date +%s) - t_net )) -ge "$NET_WAIT_MAX" ]; then
                echo "WARNING: network still unreachable after ${NET_WAIT_MAX}s; continuing anyway" >> "$LOG_FILE"
                notify_fail "network unreachable for ${NET_WAIT_MAX}s"
                break
            fi
            sleep 20
        done
        echo "--- network wait ended after $(( $(date +%s) - t_net )) s ---" >> "$LOG_FILE"
    fi
fi

# 1. Update raw data (regular season + playoff stats snapshots).
cd src/Process-Data
run_step "Get_Data"           python -m Get_Data $REFRESH_FLAG
run_step "Get_Advanced_Data"  python -m Get_Advanced_Data $REFRESH_FLAG   # CRITICAL: 28 ADV_ features for 175-feature ATS model
run_step "Get_Odds_Data"      python -m Get_Odds_Data
run_step "Get_Playoffs_Data"  python -m Get_Playoffs_Data || true
cd "$PROJECT_DIR"

# 2. Patch final scores for recent days (sbrscrape).
run_step "Patch_Scores"  python scripts/patch_scores.py

# 2a. Refresh the lines of unplayed games (non-blocking). Get_Odds_Data only inserts a
# date once, so without this the line frozen pre-tip would be the one captured a day
# or more earlier; also logs odds_history snapshots (spread_first/last/move).
echo "--- update_odds --db-only ($(date '+%H:%M:%S')) ---" >> "$LOG_FILE"
python scripts/update_odds.py --db-only >> "$LOG_FILE" 2>&1 || echo "WARNING: update_odds --db-only failed" >> "$LOG_FILE"

# 2b. Collect referee data for recent days (builds historical ATS profiles). Morning only.
if [ "$MODE" = "morning" ]; then
    t_ref=$(date +%s)
    echo "--- collect_referee_data ($(date '+%H:%M:%S')) ---" >> "$LOG_FILE"
    python scripts/collect_referee_data.py --days-back 3 >> "$LOG_FILE" 2>&1 || true
    echo "--- collect_referee_data done in $(( $(date +%s) - t_ref )) s ---" >> "$LOG_FILE"
fi

# 3. Derive playoff series state from updated odds (idempotent).
run_step "Get_Series_State" python src/Process-Data/Get_Series_State.py --all-seasons || true

# 4. Rebuild the merged dataset (now includes is_playoff + series cols).
run_step "Create_Games"  python src/Process-Data/Create_Games.py

# 4b. Leak guard (non-blocking): flag any dataset row whose stats already count its own game.
echo "--- leak_guard ($(date '+%H:%M:%S')) ---" >> "$LOG_FILE"
if ! python scripts/audit_feature_leak.py --dataset-only --seasons 2025-26 2026-27 >> "$LOG_FILE" 2>&1; then
    echo "WARNING: leak guard found dataset rows that include their own game" >> "$LOG_FILE"
    /usr/bin/osascript -e 'display notification "Leak guard: dataset rows include their own game" with title "NBA-ML Pipeline"' || true
fi

# 5. Export predictions to static JSON for Vercel.
run_step "Export_JSON"   python scripts/export_predictions.py

# 6. Git commit & push web/data/ to trigger Vercel deploy.
cd "$PROJECT_DIR"
if git diff --quiet web/data/ 2>/dev/null; then
    echo "No changes in web/data/, skipping push." >> "$LOG_FILE"
else
    git add web/data/
    COMMIT_PREFIX="daily update"
    if [ "$MODE" = "afternoon" ]; then
        COMMIT_PREFIX="afternoon update"
    fi
    git commit -m "$COMMIT_PREFIX $(date +%Y-%m-%d)

Co-Authored-By: Claude <noreply@anthropic.com>" >> "$LOG_FILE" 2>&1
    # launchd cannot unlock the macOS Keychain, so the HTTPS osxkeychain helper
    # fails with "Device not configured" (-25320). Push over SSH with a dedicated
    # passphrase-less deploy key when it exists; fall back to plain push otherwise.
    DEPLOY_KEY="$HOME/.ssh/nba-ml-deploy"
    if [ -f "$DEPLOY_KEY" ]; then
        export GIT_SSH_COMMAND="ssh -F /dev/null -i $DEPLOY_KEY -o IdentitiesOnly=yes -o BatchMode=yes -o UserKnownHostsFile=$HOME/.ssh/nba-ml-known_hosts -o StrictHostKeyChecking=accept-new"
        # Push through the named remote (pushurl override) so refs/remotes/origin/master
        # is updated; pushing to a raw URL leaves `git status` showing a phantom "ahead".
        run_step "Git_Push" git -c remote.origin.pushurl=git@github.com:freshtoastman/NBA-Machine-Learning-Sports-Betting.git push origin HEAD:master
    else
        run_step "Git_Push" git push
    fi
fi

PIPELINE_OK=1
echo "===== $(date '+%Y-%m-%d %H:%M:%S') daily_pipeline OK ($MODE) =====" >> "$LOG_FILE"
