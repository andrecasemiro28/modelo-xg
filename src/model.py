"""Modelos de xG: baseline (distância + ângulo), regressão logística completa e gradient boosting."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path(__file__).resolve().parents[1] / "data" / "processed" / "shots.csv"

NUMERIC = ["distance", "angle", "x", "y_abs", "under_pressure", "first_time", "one_on_one",
           "follows_dribble", "aerial_won", "is_female",
           "n_def_in_cone", "gk_in_cone", "gk_dist_goal", "gk_dist_shooter", "nearest_opp", "n_opp_5m"]
CATEGORICAL = ["body_part", "technique", "shot_type", "play_pattern", "assist"]
FEATURES = NUMERIC + CATEGORICAL
BASELINE = ["distance", "angle"]


def load_shots(path: Path = DATA) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"season": str})
    df = df[df["shot_type"].isin(["Open Play", "Free Kick"])].copy()  # remove 36 chutes de escanteio/saída
    df["y_abs"] = (df["y"] - 40).abs()            # distância lateral ao centro (simetria do campo)
    df["is_female"] = (df["gender"] == "female").astype(int)
    df["body_part"] = df["body_part"].replace({"Right Foot": "Foot", "Left Foot": "Foot"})
    return df.reset_index(drop=True)


def split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Divide por PARTIDA (chutes do mesmo jogo não ficam em treino e teste ao mesmo tempo)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(df, groups=df["match_id"]))
    return df.iloc[tr].copy(), df.iloc[te].copy()


def baseline_model() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])


def logistic_model() -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=30), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=3000, C=1.0))])


def boosting_model(seed: int = 42) -> Pipeline:
    pre = ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=30, sparse_output=False), CATEGORICAL),
    ])
    hgb = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=600, max_leaf_nodes=31, min_samples_leaf=80,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=30, random_state=seed)
    return Pipeline([("pre", pre), ("hgb", hgb)])


def metrics(y, p) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"log_loss": log_loss(y, p), "brier": brier_score_loss(y, p), "roc_auc": roc_auc_score(y, p),
            "xg_total/gols": p.sum() / y.sum()}


def train_all(df: pd.DataFrame):
    """Treina os 3 modelos e devolve (modelos, tabela de métricas no teste, df de teste com previsões)."""
    train, test = split(df)
    models = {
        "Baseline (distância + ângulo)": (baseline_model(), BASELINE),
        "Regressão logística": (logistic_model(), FEATURES),
        "Gradient boosting": (boosting_model(), FEATURES),
    }
    rows, fitted = [], {}
    for name, (m, cols) in models.items():
        m.fit(train[cols], train["goal"])
        test[f"p_{name}"] = m.predict_proba(test[cols])[:, 1]
        rows.append({"modelo": name, **metrics(test["goal"], test[f"p_{name}"])})
        fitted[name] = (m, cols)
    rows.append({"modelo": "xG da StatsBomb (referência)", **metrics(test["goal"], test["statsbomb_xg"])})
    return fitted, pd.DataFrame(rows).set_index("modelo"), train, test


def cross_validate(df: pd.DataFrame, n_splits: int = 5):
    """Validação cruzada agrupada por partida. Devolve métricas por fold e previsões
    fora da amostra (out-of-fold) para todos os chutes."""
    from sklearn.model_selection import GroupKFold
    oof = np.zeros(len(df))
    rows = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(df, groups=df["match_id"])):
        m = boosting_model().fit(df.iloc[tr][FEATURES], df.iloc[tr]["goal"])
        oof[te] = m.predict_proba(df.iloc[te][FEATURES])[:, 1]
        y = df.iloc[te]["goal"]
        rows.append({"fold": k + 1, **{f"modelo_{a}": b for a, b in metrics(y, oof[te]).items()},
                     **{f"statsbomb_{a}": b for a, b in metrics(y, df.iloc[te]["statsbomb_xg"]).items()}})
    return pd.DataFrame(rows), oof


def shot_scenario(x: float, y: float, body_part: str = "Foot", technique: str = "Normal",
                  shot_type: str = "Open Play", play_pattern: str = "Regular Play",
                  assist: str = "passe_curto", under_pressure: int = 0, first_time: int = 0,
                  one_on_one: int = 0, n_def_in_cone: int = 1, n_opp_5m: int = 2,
                  nearest_opp: float = 2.3, gk_off_line: float = 2.8, is_female: int = 0) -> dict:
    """Monta um chute hipotético com as mesmas features do modelo.
    O goleiro é posicionado na linha entre a bola e o centro do gol, a `gk_off_line` jardas do gol."""
    from src.extract import _in_triangle, angle, distance
    d = distance(x, y)
    t = min(gk_off_line / d, 1.0) if d > 0 else 0
    gx, gy = 120 - (120 - x) * t, 40 - (40 - y) * t
    return dict(
        distance=d, angle=angle(x, y), x=x, y_abs=abs(y - 40), under_pressure=under_pressure,
        first_time=first_time, one_on_one=one_on_one, follows_dribble=0, aerial_won=0,
        is_female=is_female, n_def_in_cone=n_def_in_cone,
        gk_in_cone=int(_in_triangle((gx, gy), (x, y), (120, 36), (120, 44))),
        gk_dist_goal=gk_off_line, gk_dist_shooter=max(d - gk_off_line, 0),
        nearest_opp=nearest_opp, n_opp_5m=n_opp_5m,
        body_part=body_part, technique=technique, shot_type=shot_type,
        play_pattern=play_pattern, assist=assist,
    )


def xg_grid(model, step: float = 1.0, **scenario) -> pd.DataFrame:
    """xG previsto em uma grade sobre o último terço do campo, para um cenário fixo."""
    xs = np.arange(80, 120, step) + step / 2
    ys = np.arange(0, 80, step) + step / 2
    rows = [shot_scenario(x, y, **scenario) for x in xs for y in ys]
    g = pd.DataFrame(rows)
    g["xg"] = model.predict_proba(g[FEATURES])[:, 1]
    g["y"] = np.tile(ys, len(xs))
    return g


def shot_density(df: pd.DataFrame, x: float, y: float, radius: float = 2.0) -> int:
    """Quantos chutes reais existem num raio de `radius` jardas do ponto (usa simetria lateral)."""
    yy = abs(y - 40)
    return int(((df["x"] - x) ** 2 + (df["y_abs"] - yy) ** 2 <= radius ** 2).sum())


def mask_sparse(grid: pd.DataFrame, df: pd.DataFrame, min_shots: int = 15, radius: float = 2.0) -> pd.DataFrame:
    """Esconde (NaN) as regiões do campo com poucos chutes reais, onde o modelo extrapola."""
    g = grid.copy()
    pts = df[["x", "y_abs"]].to_numpy()
    counts = [int((((pts[:, 0] - x) ** 2 + (pts[:, 1] - abs(y - 40)) ** 2) <= radius ** 2).sum())
              for x, y in zip(g["x"], g["y"])]
    g.loc[np.array(counts) < min_shots, "xg"] = np.nan
    return g


def finishing_table(df: pd.DataFrame, xg_col: str = "xg", min_shots: int = 50) -> pd.DataFrame:
    """Gols vs. xG por jogador, com teste de significância.

    Cada chute é tratado como uma moeda que vira gol com probabilidade p (o xG).
    Se a finalização fosse "média", o total de gols teria média sum(p) e
    variância sum(p * (1 - p)) (distribuição Poisson-binomial).
    z = (gols - xG) / desvio padrão indica quantos desvios o jogador está acima
    (ou abaixo) do esperado só por acaso: |z| > 2 dificilmente é sorte.
    """
    g = (df.assign(_var=df[xg_col] * (1 - df[xg_col]))
           .groupby("player")
           .agg(chutes=("goal", "size"), gols=("goal", "sum"), xg=(xg_col, "sum"), var=("_var", "sum")))
    g["diff"] = g["gols"] - g["xg"]
    g["desvio_padrao"] = np.sqrt(g["var"])
    g["z"] = g["diff"] / g["desvio_padrao"]
    return g[g["chutes"] >= min_shots].drop(columns="var")
