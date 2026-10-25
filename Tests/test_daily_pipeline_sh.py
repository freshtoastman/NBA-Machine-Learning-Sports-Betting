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

    def test_referee_step_logs_elapsed_seconds(self):
        src = open(SCRIPT).read()
        self.assertIn("collect_referee_data done in", src)

    def test_unknown_mode_exits_2(self):
        proc = subprocess.run([BASH, SCRIPT, "evening"], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 2)


if __name__ == "__main__":
    unittest.main()
