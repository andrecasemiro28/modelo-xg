"""Extrai todos os chutes do StatsBomb Open Data com as features do modelo de xG.

Os eventos são lidos direto do GitHub, partida por partida, sem salvar os ~12 GB brutos.
Saída: data/processed/shots.csv (uma linha por chute, pênaltis excluídos).
"""
from __future__ import annotations

import json
import math
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
OUT = Path(__file__).resolve().parents[1] / "data" / "processed" / "shots.csv"
GOAL_X, POST_L, POST_R = 120.0, 36.0, 44.0  # campo StatsBomb 120x80, gol em x=120


def _get(rel: str):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(f"{BASE_URL}/{rel}", timeout=90) as r:
                return json.loads(r.read())
        except Exception:
            if attempt == 2:
                raise


# ---------- geometria ----------
def distance(x: float, y: float) -> float:
    return math.hypot(GOAL_X - x, 40.0 - y)


def angle(x: float, y: float) -> float:
    """Ângulo (rad) de abertura do gol visto do ponto do chute."""
    a1 = math.atan2(POST_R - y, GOAL_X - x)
    a2 = math.atan2(POST_L - y, GOAL_X - x)
    return abs(a1 - a2)


def _in_triangle(p, a, b, c) -> bool:
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1, d2, d3 = sign(p, a, b), sign(p, b, c), sign(p, c, a)
    neg = d1 < 0 or d2 < 0 or d3 < 0
    pos = d1 > 0 or d2 > 0 or d3 > 0
    return not (neg and pos)


def freeze_frame_features(loc, frame) -> dict:
    """Features da posição dos jogadores no momento do chute (360 'freeze frame')."""
    if not frame:
        return dict(n_def_in_cone=None, gk_in_cone=None, gk_dist_goal=None,
                    gk_dist_shooter=None, nearest_opp=None, n_opp_5m=None)
    a, b, c = tuple(loc), (GOAL_X, POST_L), (GOAL_X, POST_R)
    opps = [p for p in frame if not p["teammate"]]
    gk = next((p for p in opps if p.get("position", {}).get("name") == "Goalkeeper"), None)
    field = [p for p in opps if p is not gk]
    dists = [math.hypot(p["location"][0] - loc[0], p["location"][1] - loc[1]) for p in opps]
    out = dict(
        n_def_in_cone=sum(_in_triangle(p["location"], a, b, c) for p in field),
        nearest_opp=min(dists) if dists else None,
        n_opp_5m=sum(d <= 5.5 for d in dists),  # 5 m ~ 5.5 jardas
        gk_in_cone=None, gk_dist_goal=None, gk_dist_shooter=None,
    )
    if gk:
        g = gk["location"]
        out.update(gk_in_cone=int(_in_triangle(g, a, b, c)),
                   gk_dist_goal=math.hypot(GOAL_X - g[0], 40 - g[1]),
                   gk_dist_shooter=math.hypot(g[0] - loc[0], g[1] - loc[1]))
    return out


def assist_type(p: dict | None) -> str:
    if p is None:
        return "sem_assistencia"
    if p.get("through_ball") or p.get("technique", {}).get("name") == "Through Ball":
        return "enfiada"
    if p.get("cut_back"):
        return "cruzamento_rasteiro"
    if p.get("cross"):
        return "cruzamento"
    if p.get("height", {}).get("name") == "High Pass":
        return "passe_alto"
    return "passe_curto"


def shots_from_match(match: dict, comp: dict) -> list[dict]:
    events = _get(f"events/{match['match_id']}.json")
    passes = {e["id"]: e["pass"] for e in events if e["type"]["name"] == "Pass"}
    rows = []
    for e in events:
        if e["type"]["name"] != "Shot" or e["period"] == 5:
            continue
        s = e["shot"]
        if s["type"]["name"] == "Penalty":
            continue
        x, y = e["location"][:2]
        row = dict(
            match_id=match["match_id"],
            competition=comp["competition_name"], season=comp["season_name"],
            gender=comp["competition_gender"],
            player=e["player"]["name"], team=e["team"]["name"],
            minute=e["minute"], x=x, y=y,
            distance=distance(x, y), angle=angle(x, y),
            body_part=s["body_part"]["name"],
            technique=s.get("technique", {}).get("name", "Normal"),
            shot_type=s["type"]["name"],              # Open Play / Free Kick / Corner
            play_pattern=e["play_pattern"]["name"],
            under_pressure=int(bool(e.get("under_pressure"))),
            first_time=int(bool(s.get("first_time"))),
            one_on_one=int(bool(s.get("one_on_one"))),
            follows_dribble=int(bool(s.get("follows_dribble"))),
            aerial_won=int(bool(s.get("aerial_won"))),
            assist=assist_type(passes.get(s.get("key_pass_id"))),
            statsbomb_xg=s["statsbomb_xg"],
            goal=int(s["outcome"]["name"] == "Goal"),
        )
        row.update(freeze_frame_features(e["location"], s.get("freeze_frame")))
        rows.append(row)
    return rows


def build(workers: int = 16) -> pd.DataFrame:
    jobs = []
    for comp in _get("competitions.json"):
        for m in _get(f"matches/{comp['competition_id']}/{comp['season_id']}.json"):
            jobs.append((m, comp))
    print(f"{len(jobs)} partidas")
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, r in enumerate(pool.map(lambda j: shots_from_match(*j), jobs)):
            rows.extend(r)
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)} partidas · {len(rows)} chutes", flush=True)
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(OUT, index=False)
    return df


if __name__ == "__main__":
    d = build()
    print(f"{len(d)} chutes salvos em {OUT}")
