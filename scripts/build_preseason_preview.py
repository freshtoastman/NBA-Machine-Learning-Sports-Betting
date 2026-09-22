#!/usr/bin/env python3
"""Build the 2026-27 preseason team preview.

Merges (a) last-season per-team results computed from the local SQLite data
with (b) hand-curated offseason facts (trades, signings, win totals, title
odds, injuries, coaching changes) into:

  web/data/preseason_2026-27.json   -> consumed by web/app.py (offseason section)
  docs/preseason_2026-27_team_preview.md -> long-form analysis

Usage:
  python scripts/build_preseason_preview.py --summary <team_summary.json> --curated <offseason_curated.py>

The curated module is a plain Python file defining TITLE_ODDS, WIN_TOTAL,
NEW_COACH, CONF and TEAMS (see docs/preseason_2026-27_team_preview.md for
sources). Re-run whenever lines move or rosters change before opening night.
"""
from __future__ import annotations

import argparse
import json
import runpy
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.Utils.Teams import team_logo_url, team_name_zh  # noqa: E402

SEASON = "2026-27"
PREV = "2025-26"

# Roster churn: how much of last season's on-court identity changed. Drives the
# "features stale" warning for the first ~15 games; the ATS model's team
# features are rolling averages of last season until enough new games exist.
CHURN = {
    "high": ["Philadelphia 76ers", "Miami Heat", "Milwaukee Bucks", "Minnesota Timberwolves",
             "Charlotte Hornets", "LA Clippers", "Toronto Raptors", "Portland Trail Blazers",
             "Los Angeles Lakers", "Washington Wizards", "Boston Celtics", "Memphis Grizzlies"],
    "medium": ["Atlanta Hawks", "Oklahoma City Thunder", "Cleveland Cavaliers", "Dallas Mavericks",
               "Brooklyn Nets", "Chicago Bulls", "Utah Jazz", "Detroit Pistons", "Phoenix Suns",
               "Indiana Pacers", "Houston Rockets", "Golden State Warriors"],
}


def churn_of(team: str) -> str:
    for lvl, teams in CHURN.items():
        if team in teams:
            return lvl
    return "low"


def tier_of(win_total: float) -> str:
    if win_total >= 55:
        return "冠軍熱門"
    if win_total >= 47:
        return "分區前段"
    if win_total >= 40:
        return "季後賽邊緣"
    if win_total >= 33:
        return "附加賽以下"
    return "重建 / 擺爛"


def pct(rec: str) -> float | None:
    """'43-0-39' -> cover% excluding pushes."""
    try:
        w, p, l = (int(x) for x in rec.split("-"))
    except (ValueError, AttributeError):
        return None
    return round(w / (w + l), 3) if (w + l) else None


PLAYOFF_ZH = {
    "NBA Champion": "總冠軍",
    "Eliminated in NBA Finals": "總決賽敗",
    "Eliminated in Conference Finals": "分區決賽敗",
    "Eliminated in Conference Semifinals": "分區準決賽敗",
    "Eliminated in First Round": "首輪敗",
    "Missed playoffs (lost in play-in)": "附加賽出局",
    "Missed playoffs": "未進季後賽",
}


def wins_of(rec: str) -> int:
    return int(rec.split("-")[0])


def build(summary: dict, cur: dict) -> dict:
    teams_out = {}
    for team, c in cur["TEAMS"].items():
        s = summary["teams"][team]
        rs, po = s["regular_season"], s.get("playoffs")
        adv = s.get("advanced_regular_season_nba_com") or {}
        wt = cur["WIN_TOTAL"][team]
        last_w = wins_of(rs["record"])
        teams_out[team] = {
            "team": team,
            "team_zh": team_name_zh(team),
            "logo": team_logo_url(team),
            "conference": "East" if team in cur["CONF"]["East"] else "West",
            "last_season": {
                "record": rs["record"],
                "home_record": rs["home_record"],
                "away_record": rs["away_record"],
                "point_diff": rs["point_diff_per_game"],
                "net_rating": adv.get("NET_RATING"),
                "off_rating": adv.get("OFF_RATING"),
                "def_rating": adv.get("DEF_RATING"),
                "net_rank": adv.get("NET_RATING_RANK"),
                "ats": rs["ats"]["overall"],
                "ats_cover_pct": pct(rs["ats"]["overall"]),
                "ats_home": rs["ats"]["home"],
                "ats_home_pct": pct(rs["ats"]["home"]),
                "ats_away": rs["ats"]["away"],
                "ats_away_pct": pct(rs["ats"]["away"]),
                "ou": rs["over_under"]["record"],
                "over_pct": rs["over_under"]["over_pct"],
                "playoff_result": PLAYOFF_ZH.get(s.get("playoff_round_reached") or "", s.get("playoff_round_reached") or "未進季後賽"),
                "playoff_record": po["record"] if po else None,
                "playoff_ats": po["ats"]["overall"] if po else None,
            },
            "market": {
                "win_total": wt,
                "win_total_delta": round(wt - last_w, 1),
                "title_odds": cur["TITLE_ODDS"].get(team),
                "tier": tier_of(wt),
            },
            "offseason": {
                "grade_espn": c["grade"][0],
                "grade_cbs": c["grade"][1],
                "adds": c["adds"],
                "departs": c["departs"],
                "injury": c["injury"],
                "new_coach": cur["NEW_COACH"].get(team),
                "roster_churn": churn_of(team),
            },
            "outlook": c["outlook"],
            "ats_angle": c["ats_angle"],
        }

    league = summary["league"]["regular_season"]
    order = sorted(teams_out, key=lambda t: -teams_out[t]["market"]["win_total"])
    return {
        "season": SEASON,
        "prev_season": PREV,
        "generated": date.today().isoformat(),
        "sources": [
            "Local: Data/OddsData.sqlite '2025-26' (closing spreads), AdvancedTeamData.sqlite 2026-05-28",
            "NBA.com offseason deals tracker (2026)", "ESPN / CBS Sports offseason grades (2026)",
            "BetMGM 2026-27 win totals (via Yahoo Sports)", "ESPN 2027 championship odds",
            "ESPN 2026-27 injury returns", "NBA.com 2026 coaching tracker",
        ],
        "league_last_season": {
            "home_cover_pct": league["home_cover_rate_excl_push"],
            "favorite_cover_pct": league["favorite_cover_rate_excl_push"],
            "home_win_pct": league["home_win_pct"],
            "over_pct": league["over_rate_excl_push"],
            "avg_home_spread": league["avg_home_spread"],
            "games": league["games"],
        },
        "order_by_win_total": order,
        "teams": teams_out,
    }


def fmt_pct(v):
    return "—" if v is None else f"{v * 100:.1f}%"


def write_md(data: dict, path: Path) -> None:
    T = data["teams"]
    L = data["league_last_season"]
    lines = []
    A = lines.append
    A(f"# {SEASON} 開季前瞻：各隊狀況分析")
    A("")
    A(f"> 產出日期：{data['generated']}（開季 2026-10-20 前 4 週）  ")
    A(f"> 資料：上季 {PREV} 本地賽果與收盤盤口 + 2026 休賽季異動、勝場盤（BetMGM）、冠軍賠率、傷兵、教練異動  ")
    A("> 定位：作為 ATS 模型開季前 15 場的「特徵過時」提醒與市場定價落差清單，不是選隊指南。")
    A("")
    A("## 1. 上季聯盟基準（模型校準用）")
    A("")
    A("| 指標 | 2025-26 常規賽 | 解讀 |")
    A("|---|---|---|")
    A(f"| 主隊勝率 | {L['home_win_pct']*100:.1f}% | 主場優勢正常 |")
    A(f"| 主隊過盤率 | {L['home_cover_pct']*100:.1f}% | 市場對主場略微低估（>50%） |")
    A(f"| 讓分方過盤率 | {L['favorite_cover_pct']*100:.1f}% | 受讓方略優，符合模型 away≥9% 較嚴門檻 |")
    A(f"| 大分過盤率 | {L['over_pct']*100:.1f}% | 總分盤幾乎完全定價 |")
    A(f"| 平均主場讓分 | +{L['avg_home_spread']} | |")
    A("")
    A("## 2. 休賽季大局")
    A("")
    A("- **LeBron James → PHI、Jaylen Brown → PHI**：76 人從 60-1 變 +900，與衛冕軍 NYK 並列東區前二。")
    A("- **Giannis → MIA**（6/22）：熱火付出 Herro / Jaquez / Ware / Jakučionis；公鹿失去核心，勝場盤只剩 26.5。")
    A("- **Kawhi → TOR、LaMelo → MIN、Ja Morant → POR、Walker Kessler → LAL**：四筆換核心交易，全部造成模型特徵過時。")
    A("- **OKC 為脫離第二豪華稅線送走 Dort / Wiggins / Joe**，深度下降但仍是共同冠軍熱門 +270（與 SAS 並列）。")
    A("- **六隊換教練**：MIL（Taylor Jenkins）、POR（Micah Nori）、DAL（Dusty May）、CHI（Tiago Splitter）、NOP（Jamahl Mosley）、ORL（Sean Sweeney）。")
    A("- **重要傷兵回歸**：Haliburton（IND）、Lillard（POR）、Kyrie（DAL）、VanVleet（HOU）、Dejounte Murray（NOP）；Jimmy Butler（GSW）最快 1–2 月。")
    A("")
    A("## 3. 市場分層（BetMGM 勝場盤）")
    A("")
    A("| 隊伍 | 勝場盤 | 上季戰績 | 差距 | 冠軍賠率 | 分層 |")
    A("|---|---|---|---|---|---|")
    for t in data["order_by_win_total"]:
        d = T[t]
        m, ls = d["market"], d["last_season"]
        delta = m["win_total_delta"]
        A(f"| {d['team_zh']} | {m['win_total']} | {ls['record']} | {delta:+.1f} | {m['title_odds'] or '—'} | {m['tier']} |")
    A("")
    A("### 3a. 市場相對上季調整最大的球隊")
    A("")
    ups = sorted(T.values(), key=lambda d: -d["market"]["win_total_delta"])[:6]
    downs = sorted(T.values(), key=lambda d: d["market"]["win_total_delta"])[:6]
    A("**上修**（市場預期比上季勝場多）：")
    for d in ups:
        A(f"- {d['team_zh']} {d['market']['win_total_delta']:+.1f}：{d['outlook'].split('；')[0].split('。')[0]}")
    A("")
    A("**下修**：")
    for d in downs:
        A(f"- {d['team_zh']} {d['market']['win_total_delta']:+.1f}：{d['outlook'].split('；')[0].split('。')[0]}")
    A("")
    A("## 4. 上季 ATS 側寫（哪些隊市場定價一直錯）")
    A("")
    A("| 隊伍 | ATS 總 | 過盤率 | 主場 ATS | 客場 ATS | 大小分 | 判讀 |")
    A("|---|---|---|---|---|---|---|")
    by_ats = sorted(T.values(), key=lambda d: -(d["last_season"]["ats_cover_pct"] or 0))
    for d in by_ats:
        ls = d["last_season"]
        hp, ap = ls["ats_home_pct"] or 0, ls["ats_away_pct"] or 0
        if (ls["ats_cover_pct"] or 0) >= 0.56:
            note = "市場長期低估"
        elif (ls["ats_cover_pct"] or 0) <= 0.44:
            note = "市場長期高估"
        elif hp - ap >= 0.15:
            note = "主場被低估、客場被高估"
        elif ap - hp >= 0.15:
            note = "客場被低估、主場被高估"
        else:
            note = "定價大致準確"
        A(f"| {d['team_zh']} | {ls['ats']} | {fmt_pct(ls['ats_cover_pct'])} | {ls['ats_home']} | {ls['ats_away']} | {ls['ou']} | {note} |")
    A("")
    A("重點：")
    A("- **OKC 64 勝卻 39-42 ATS**：市場給冠軍隊的讓分長期過深，主場 19-23。若本季深度下降後市場仍照 60.5 勝定價，OKC 讓分方在 10–11 月是最值得反向觀察的對象。")
    A("- **CHA 51-31 ATS（聯盟最佳）**：上季被低估最嚴重的球隊，但核心 LaMelo 已走，這個側寫不能直接延續。")
    A("- **NYK 主場 28-12 / 客場 15-27**：衛冕軍主客場 ATS 差距 32 個百分點，模型的主客非對稱門檻在 NYK 身上會特別有效。")
    A("- **CLE 32-1-49**：常規賽 52 勝但過盤率 39.5%，市場一直高估；休賽季沒補強，此側寫很可能延續。")
    A("- **POR 主場 27-14、NOP 主場 25-15**：弱隊主場受讓過盤率高，是 home≥8% 信號在弱隊端的來源。")
    A("- **BOS 小分 51-31、DEN 大分 52-30、UTA 大分 51-31**：總分盤側寫最極端的三隊；BOS 節奏聯盟最慢（95.6），DEN / UTA 失分結構未改。")
    A("")
    A("## 5. 對 ATS 模型的具體建議（開季前 15 場）")
    A("")
    A("模型的球隊特徵（滾動淨效率、ADV_ 進階數據）在新賽季前 10–15 場仍以上季為主。以下球隊陣容變動太大，上季特徵已不代表本季實力：")
    A("")
    high = [d for d in T.values() if d["offseason"]["roster_churn"] == "high"]
    A("| 隊伍 | 變動 | 特徵偏差方向 |")
    A("|---|---|---|")
    bias = {
        "Philadelphia 76ers": "上季 45 勝 → 本季應為 50+，模型會**低估** PHI",
        "Miami Heat": "上季 43 勝，加 Giannis 後模型會**低估** MIA 常規賽",
        "Milwaukee Bucks": "上季 32 勝含半季 Giannis，模型會**高估** MIL",
        "Minnesota Timberwolves": "換掉 Randle / Reid，方向不明，信號降權",
        "Charlotte Hornets": "上季 ATS 最佳但失去 LaMelo，模型會**高估** CHA",
        "LA Clippers": "失去 Kawhi 進入重建，模型會**高估** LAC",
        "Toronto Raptors": "加 Kawhi（若交易完成），模型會**低估** TOR；輪休場次需即時抓",
        "Portland Trail Blazers": "Morant + Lillard 加入但兼容性未知，信號降權",
        "Los Angeles Lakers": "失去 LeBron，模型會**高估** LAL",
        "Washington Wizards": "17 勝隊加 Trae / AD / Dybantsa / Ayton，模型會**嚴重高估對手讓分**",
        "Boston Celtics": "Brown → George，方向略負，信號降權",
        "Memphis Grizzlies": "失去 Morant 但上季 Morant 只打 20 場，偏差有限",
    }
    for d in high:
        A(f"| {d['team_zh']} | {d['offseason']['departs'][:40]}… → {d['offseason']['adds'][:40]}… | {bias.get(d['team'], '')} |")
    A("")
    A("**執行規則（建議寫入模型的季初抑制）：**")
    A("1. 10/20–11/10：涉及上表球隊的 ATS 信號，edge 門檻由 8%/9% 提高到 12%/13%；或直接列為觀察不下注。")
    A("2. 每隊累積 12 場新賽季比賽後，滾動特徵才以本季為主，恢復正常門檻。")
    A("3. PHI / MIA / TOR 的明星輪休（Embiid、Giannis、Kawhi）是盤口異常移動的主要來源，維持「線移 ≥2 分才視為訊號」的 CLV 規則。")
    A("4. 擺爛梯隊（SAC、BKN、MIL、CHI、MEM、LAC、NOP）3 月後信號全部抑制，與上季規則一致。")
    A("")
    A("## 6. 各隊逐一分析")
    A("")
    for conf, zh in (("East", "東區"), ("West", "西區")):
        A(f"### {zh}")
        A("")
        for t in data["order_by_win_total"]:
            d = T[t]
            if d["conference"] != conf:
                continue
            ls, m, o = d["last_season"], d["market"], d["offseason"]
            A(f"#### {d['team_zh']}（{t}）")
            A("")
            A(f"- **上季**：{ls['record']}（主 {ls['home_record']} / 客 {ls['away_record']}），淨效率 {ls['net_rating']:+.1f}（第 {ls['net_rank']}），ATS {ls['ats']}（主 {ls['ats_home']} / 客 {ls['ats_away']}），{ls['playoff_result']}" + (f"，季後賽 {ls['playoff_record']}、ATS {ls['playoff_ats']}" if ls["playoff_record"] else "") + "。")
            A(f"- **市場**：勝場盤 {m['win_total']}（相對上季 {m['win_total_delta']:+.1f}）" + (f"，冠軍賠率 {m['title_odds']}" if m["title_odds"] else "") + f"，分層「{m['tier']}」。")
            A(f"- **休賽季評分**：ESPN {o['grade_espn'] or '—'} / CBS {o['grade_cbs'] or '—'}；陣容變動：{o['roster_churn']}" + (f"；新教練 {o['new_coach']}" if o["new_coach"] else "") + "。")
            A(f"- **補進**：{o['adds']}")
            A(f"- **流失**：{o['departs']}")
            if o["injury"]:
                A(f"- **傷兵**：{o['injury']}")
            A(f"- **狀況**：{d['outlook']}")
            A(f"- **ATS 角度**：{d['ats_angle']}")
            A("")
    A("---")
    A("")
    A("資料來源：NBA.com 休賽季交易追蹤、ESPN / CBS Sports 休賽季評分、Yahoo Sports（BetMGM 勝場盤）、ESPN 2027 冠軍賠率、ESPN 傷兵回歸專題、NBA.com 教練異動追蹤；上季數據來自本專案 Data/OddsData.sqlite 與 AdvancedTeamData.sqlite。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--curated", required=True)
    ap.add_argument("--json-out", default=str(ROOT / "web/data/preseason_2026-27.json"))
    ap.add_argument("--md-out", default=str(ROOT / "docs/preseason_2026-27_team_preview.md"))
    a = ap.parse_args()
    summary = json.loads(Path(a.summary).read_text(encoding="utf-8"))
    cur = runpy.run_path(a.curated)
    data = build(summary, cur)
    Path(a.json_out).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    write_md(data, Path(a.md_out))
    print(f"wrote {a.json_out} ({len(data['teams'])} teams) and {a.md_out}")


if __name__ == "__main__":
    main()
