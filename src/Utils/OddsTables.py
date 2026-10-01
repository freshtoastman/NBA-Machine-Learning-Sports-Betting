"""Single source of truth for reading a season's games out of OddsData.sqlite.

Several table-name conventions exist for the same season (`odds_2025-26`,
`2025-26`, `odds_2025-26_new`, ...). Every consumer used to carry its own
"freshest table" picker keyed on MAX(Date); a partial table that happened to
hold the latest date then silently replaced the complete one (2025-26 lost
~760 games that way). `load_season_odds` merges every candidate table instead,
so no single stale table can win.
"""
import re

import pandas as pd

ODDS_COLUMNS = [
    "Date", "Home", "Away", "OU", "Spread", "ML_Home", "ML_Away",
    "Points", "Win_Margin", "Days_Rest_Home", "Days_Rest_Away",
]
_NUMERIC_COLUMNS = ODDS_COLUMNS[3:]
_KEY = ["Date", "Home", "Away"]
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([ T].*)?$")

# Same franchise under two spellings; only used to build the de-dup key.
TEAM_ALIASES = {"Los Angeles Clippers": "LA Clippers"}


def canonical_team(name):
    return TEAM_ALIASES.get(name, name)


def season_odds_tables(con, season_key):
    """Existing candidate tables for a season, in legacy priority order."""
    candidates = [
        f"odds_{season_key}_new",
        f"odds_{season_key}",
        f"{season_key}_new",
        f"{season_key}",
    ]
    return [
        t for t in candidates
        if con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (t,)
        ).fetchone()
    ]


def _read_table(con, table):
    try:
        df = pd.read_sql_query(f'SELECT * FROM "{table}"', con)
    except Exception:
        return pd.DataFrame(columns=ODDS_COLUMNS)
    if not set(_KEY).issubset(df.columns):
        return pd.DataFrame(columns=ODDS_COLUMNS)
    df = df.reindex(columns=ODDS_COLUMNS)
    df = df.dropna(subset=_KEY)
    # Legacy tables carry unusable dates such as '2007-08-0102'; drop those rows.
    dates = df["Date"].astype(str)
    df = df[dates.map(lambda d: bool(_DATE_RE.match(d)))].copy()
    df["Date"] = df["Date"].astype(str).str[:10]
    for col in _NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def load_season_odds(con, season_key, columns=None):
    """Return one row per game for the season, merged across all candidate tables.

    The table with the most usable rows is the base. Rows from the other tables
    are added only for games the base lacks; when the same game appears twice,
    the row with a final score wins, then the base table's row. Dates come back
    as 'YYYY-MM-DD' strings, numeric columns as floats (NaN when missing).
    Returns an empty frame when the season has no table.
    """
    frames = [(t, _read_table(con, t)) for t in season_odds_tables(con, season_key)]
    frames = [(t, df) for t, df in frames if not df.empty]
    if not frames:
        out = pd.DataFrame(columns=ODDS_COLUMNS)
        return out[columns] if columns else out

    # Stable sort: ties keep the legacy priority order.
    frames.sort(key=lambda item: len(item[1]), reverse=True)
    parts = []
    for rank, (_, df) in enumerate(frames):
        df = df.copy()
        df["_rank"] = rank
        df["_order"] = range(len(df))
        parts.append(df)
    merged = pd.concat(parts, ignore_index=True)
    merged["_home"] = merged["Home"].map(canonical_team)
    merged["_away"] = merged["Away"].map(canonical_team)
    merged["_scored"] = (merged["Points"].fillna(0) > 0).astype(int)

    if len(frames) > 1:
        # A secondary-only row one day off a base row for the same pairing is a
        # date-shifted copy of that game, not a new one.
        base = merged[merged["_rank"] == 0]
        base_keys = set(zip(base["Date"], base["_home"], base["_away"]))
        base_pair_days = {}
        for d, h, a in base_keys:
            base_pair_days.setdefault(frozenset((h, a)), set()).add(pd.Timestamp(d))

        def shifted_copy(row):
            if row["_rank"] == 0 or (row["Date"], row["_home"], row["_away"]) in base_keys:
                return False
            days = base_pair_days.get(frozenset((row["_home"], row["_away"])), ())
            day = pd.Timestamp(row["Date"])
            return any(abs((day - other).days) == 1 for other in days)

        merged = merged[~merged.apply(shifted_copy, axis=1)]

    merged = merged.sort_values(
        ["_scored", "_rank", "_order"], ascending=[False, True, True], kind="mergesort"
    )
    merged = merged.drop_duplicates(subset=["Date", "_home", "_away"], keep="first")
    merged = merged.sort_values(["Date", "_rank", "_order"], kind="mergesort")
    out = merged[ODDS_COLUMNS].reset_index(drop=True)
    return out[columns] if columns else out


def load_seasons_odds(con, season_keys, columns=None):
    """Concatenate `load_season_odds` over several seasons (missing seasons skipped)."""
    frames = [load_season_odds(con, s, columns) for s in season_keys]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=columns or ODDS_COLUMNS)
    return pd.concat(frames, ignore_index=True)
