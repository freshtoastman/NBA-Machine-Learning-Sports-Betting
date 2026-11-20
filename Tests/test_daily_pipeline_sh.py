"""Regression tests for scripts/daily_pipeline.sh under the shell launchd uses.

macOS ships bash 3.2, where ``set -u`` makes ``"${EMPTY_ARRAY[@]}"`` a fatal
"unbound variable" (fixed only in bash 4.4). The 2026-10-14 06:00 run died on
exactly that line before its first step, leaving no FAILED marker in the log.
"""
import os
import re
import subprocess
import tempfile
import unittest

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "daily_pipeline.sh")
BASH = "/bin/bash"


def _prologue():
    """The script up to (not including) the first `cd "$PROJECT_DIR"` (mode parsing,
    log setup, notify/exit trap) plus the run_step function, so the control flow runs
    outside the project: no venv, no network, no osascript."""
    src = open(SCRIPT).read()
    head = src.split('cd "$PROJECT_DIR"', 1)[0]
    run_step = re.search(r"^run_step\(\) \{.*?^\}\n", src, re.S | re.M).group(0)
    return head + "\n" + run_step + "\n"


class DailyPipelineShellTest(unittest.TestCase):
    def test_syntax_under_bash_3_2(self):
        subprocess.run([BASH, "-n", SCRIPT], check=True)

    def test_no_unguarded_empty_array_expansion(self):
        # Any "${NAME[@]}" is fatal under bash 3.2 + set -u when NAME is empty.
        code = "\n".join(l for l in open(SCRIPT) if not l.lstrip().startswith("#"))
        unguarded = re.findall(r'"\$\{[A-Z_]+\[@\]\}"', code)
        self.assertEqual(unguarded, [], f"bash 3.2 + set -u unsafe expansions: {unguarded}")

    def _run_mode(self, mode):
        head = _prologue()
        with tempfile.TemporaryDirectory() as tmp:
            harness = (
                head
                + '\nLOG_FILE="$LOG_DIR/test.log"\n'
                + 'notify_fail() { echo "NOTIFY $1" >> "$LOG_FILE"; }\n'
                + 'echo "GET_DATA_ARGS=[$REFRESH_FLAG]" >> "$LOG_FILE"\n'
                + "PIPELINE_OK=1\n"
            )
            harness = harness.replace('LOG_DIR="$HOME/Library/Logs/nba-ml"', f'LOG_DIR="{tmp}"')
            proc = subprocess.run(
                [BASH, "-c", harness, "daily_pipeline.sh", mode],
                capture_output=True, text=True, env={**os.environ, "HOME": tmp, "NBA_ML_CAFFEINATED": "1"},
            )
            log = open(os.path.join(tmp, "test.log")).read() if os.path.exists(os.path.join(tmp, "test.log")) else ""
            monthly = [f for f in os.listdir(tmp) if f.startswith("pipeline-")]
            start = open(os.path.join(tmp, monthly[0])).read() if monthly else ""
            return proc, start + log

    def test_morning_mode_passes_no_refresh_flag(self):
        proc, log = self._run_mode("morning")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn("unbound variable", proc.stderr)
        self.assertIn("GET_DATA_ARGS=[]", log)
        self.assertIn("start (morning)", log)

    def test_afternoon_mode_passes_refresh_flag(self):
        proc, log = self._run_mode("afternoon")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("GET_DATA_ARGS=[--refresh-latest]", log)

    def test_unexpected_abort_writes_failed_marker(self):
        head = _prologue()
        with tempfile.TemporaryDirectory() as tmp:
            harness = (
                head.replace('LOG_DIR="$HOME/Library/Logs/nba-ml"', f'LOG_DIR="{tmp}"')
                + '\nLOG_FILE="$LOG_DIR/test.log"\n'
                + 'notify_fail() { echo "NOTIFY $1" >> "$LOG_FILE"; }\n'
                + 'echo "$UNSET_VARIABLE_FOR_TEST"\n'   # set -u abort outside run_step
            )
            proc = subprocess.run([BASH, "-c", harness, "daily_pipeline.sh", "morning"],
                                  capture_output=True, text=True, env={**os.environ, "HOME": tmp, "NBA_ML_CAFFEINATED": "1"})
            log = open(os.path.join(tmp, "test.log")).read()
        self.assertNotEqual(proc.returncode, 0, "bash 3.2 EXIT trap must not turn a set -u abort into exit 0")
        self.assertIn("FAILED (exit", log)
        self.assertIn("NOTIFY unexpected abort", log)

    def test_run_step_failure_keeps_exit_1_and_single_marker(self):
        head = _prologue()
        with tempfile.TemporaryDirectory() as tmp:
            harness = (
                head.replace('LOG_DIR="$HOME/Library/Logs/nba-ml"', f'LOG_DIR="{tmp}"')
                + '\nLOG_FILE="$LOG_DIR/test.log"\n'
                + 'notify_fail() { echo "NOTIFY $1" >> "$LOG_FILE"; }\n'
                + 'run_step "Boom" false\n'
            )
            proc = subprocess.run([BASH, "-c", harness, "daily_pipeline.sh", "morning"],
                                  capture_output=True, text=True, env={**os.environ, "HOME": tmp, "NBA_ML_CAFFEINATED": "1"})
            log = open(os.path.join(tmp, "test.log")).read()
        self.assertEqual(proc.returncode, 1)
        self.assertIn("FAILED at Boom", log)
        self.assertNotIn("unexpected abort", log)
        self.assertEqual(log.count("FAILED"), 1)

    def test_reexecs_under_caffeinate_once(self):
        # 2026-10-19: the run was frozen between dark wakes for 2 h 54 min. The script
        # must hold a no-idle-sleep assertion for its whole life, and must not loop.
        src = open(SCRIPT).read()
        self.assertIn('exec /usr/bin/caffeinate -i -m -s /bin/bash "$0" "$@"', src)
        self.assertIn('[ -z "${NBA_ML_CAFFEINATED:-}" ]', src)
        self.assertIn("export NBA_ML_CAFFEINATED=1", src)
        guard = src.index('NBA_ML_CAFFEINATED:-'); first_step = src.index('run_step "Get_Data"')
        self.assertLess(guard, first_step, "caffeinate guard must precede the first step")

    def test_push_ssh_drops_dead_connection(self):
        # 2026-11-16 06:00: a battery dark-wake left the deploy-key push hanging on a
        # dead TCP session for 6 h 4 min before the kernel timed it out. Keepalives
        # bound the hang to ~60 s so the FAILED marker lands and the 15:00 run can push.
        src = open(SCRIPT).read()
        m = re.search(r'export GIT_SSH_COMMAND="(.*?)"\n', src)
        self.assertIsNotNone(m)
        for opt in ("ConnectTimeout=30", "ServerAliveInterval=15", "ServerAliveCountMax=4"):
            self.assertIn(f"-o {opt}", m.group(1))

    def _power_log(self, mode, power, extra_env=None):
        with tempfile.TemporaryDirectory() as home:
            env = dict(os.environ, HOME=home, NBA_ML_CAFFEINATED="1", NBA_ML_POWER_SOURCE=power)
            env.update(extra_env or {})
            harness = _prologue() + 'echo "LEAN=$BATTERY_LEAN"\nPIPELINE_OK=1\n'
            proc = subprocess.run([BASH, "-c", harness, "daily_pipeline.sh", mode],
                                  capture_output=True, text=True, env=env)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            logs = os.path.join(home, "Library", "Logs", "nba-ml")
            log = open(os.path.join(logs, os.listdir(logs)[0])).read()
            return proc.stdout, log

    def test_battery_morning_runs_lean(self):
        # 2026-11-16: a battery morning took 72 min to reach Export_JSON and two games
        # froze on the wrong line. Referee + Create_Games are skipped there only.
        out, log = self._power_log("morning", "'Battery Power'")
        self.assertIn("LEAN=1", out)
        self.assertIn("battery morning: lean run", log)
        out, log = self._power_log("morning", "'AC Power'")
        self.assertIn("LEAN=0", out)
        self.assertNotIn("lean run", log)
        out, _ = self._power_log("afternoon", "'Battery Power'")
        self.assertIn("LEAN=0", out)
        out, _ = self._power_log("morning", "'Battery Power'", {"NBA_ML_BATTERY_LEAN": "0"})
        self.assertIn("LEAN=0", out)

    def test_lean_run_guards_the_slow_steps_only(self):
        src = open(SCRIPT).read()
        self.assertIn('[ "$MODE" = "morning" ] && [ "$BATTERY_LEAN" -eq 0 ]', src)
        guard = src.index('if [ "$BATTERY_LEAN" -eq 0 ]; then\n    # 4. Rebuild')
        self.assertLess(guard, src.index('run_step "Create_Games"'))
        export = src[src.index("# 5. Export predictions"):]
        self.assertTrue(export.split("\n")[1].startswith('run_step "Export_JSON"'))
        for step in ("Patch_Scores", "Get_Odds_Data"):
            line = [l for l in src.split("\n") if f'run_step "{step}"' in l][0]
            self.assertFalse(line.startswith(" "), f"{step} must not be skipped")

    def test_lean_run_freezes_before_any_other_fetch(self):
        """Battery morning: the line refresh + export that lock the pre-game block
        must run before Get_Data / Get_Advanced_Data / Get_Odds_Data (2026-11-20:
        the export beat four tip-offs by 78 seconds)."""
        with open(SCRIPT, encoding="utf-8") as fh:
            src = fh.read()
        ff = src.index('if [ "$BATTERY_LEAN" -eq 1 ]; then')
        block = src[ff:src.index("fi\n", ff)]
        self.assertLess(block.index("update_odds.py --db-only"), block.index("export_predictions.py"))
        self.assertLess(ff, src.index('run_step "Get_Data"'))
        self.assertGreater(ff, src.index("net_up; do"))        # after the network wait
        self.assertNotIn("run_step", block)                     # never fatal

    def test_referee_step_logs_elapsed_seconds(self):
        src = open(SCRIPT).read()
        self.assertIn("collect_referee_data done in", src)

    def test_unknown_mode_exits_2(self):
        proc = subprocess.run([BASH, SCRIPT, "evening"], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 2)


if __name__ == "__main__":
    unittest.main()
