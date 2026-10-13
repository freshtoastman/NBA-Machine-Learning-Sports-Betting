# launchd jobs

Two user agents run `scripts/daily_pipeline.sh` (Taipei time):

| Label | Time | Mode | Purpose |
|---|---|---|---|
| `com.lchao.nba-ml.daily` | 06:00 | morning | 18:00 ET previous day: new stats table, scores, odds, export, push |
| `com.lchao.nba-ml.afternoon` | 15:00 | afternoon | 03:00 ET: re-fetch today's stats table (`--refresh-latest`, only before 12:00 ET) so pre-game features include every game through yesterday; refresh lines; re-export |

Install / reload:

```bash
cp scripts/launchd/*.plist ~/Library/LaunchAgents/
launchctl unload ~/Library/LaunchAgents/com.lchao.nba-ml.afternoon.plist 2>/dev/null
launchctl load ~/Library/LaunchAgents/com.lchao.nba-ml.afternoon.plist
```

Logs: `~/Library/Logs/nba-ml/pipeline-YYYY-MM.log` (both modes, each run tagged `(morning)` / `(afternoon)`).
