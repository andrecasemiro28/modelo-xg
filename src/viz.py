"""Gráficos do projeto de xG (paleta sóbria: um azul de destaque + cinza)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

ACCENT, MUTED, TEXT, SUB, BG = "#1f5fa8", "#9aa3ad", "#1f2328", "#57606a", "#f8f7f4"

FEATURE_LABELS = {
    "distance": "Distância ao gol", "angle": "Ângulo do gol", "x": "Posição (profundidade)",
    "y_abs": "Posição (lateral)", "under_pressure": "Sob pressão", "first_time": "Chute de primeira",
    "one_on_one": "Mano a mano com goleiro", "follows_dribble": "Após drible", "aerial_won": "Disputa aérea ganha",
    "is_female": "Futebol feminino", "n_def_in_cone": "Defensores entre bola e gol",
    "gk_in_cone": "Goleiro no ângulo", "gk_dist_goal": "Goleiro adiantado", "gk_dist_shooter": "Distância goleiro–bola",
    "nearest_opp": "Marcador mais próximo", "n_opp_5m": "Adversários a até 5 m",
    "body_part": "Parte do corpo", "technique": "Técnica", "shot_type": "Tipo (bola rolando/falta)",
    "play_pattern": "Origem da jogada", "assist": "Tipo de assistência",
}


def _style(ax, title, subtitle=None):
    ax.set_title(title, loc="left", size=14, weight="bold", color=TEXT, pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, size=10, color=SUB)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def draw_final_third(ax, color="#b9bec5", lw=1.2):
    """Desenha o último terço do campo (vertical, gol no topo) em coordenadas StatsBomb."""
    kw = dict(color=color, lw=lw, zorder=3)
    ax.plot([0, 80, 80, 0, 0], [80, 80, 120, 120, 80], **kw)            # contorno
    ax.plot([18, 18, 62, 62], [120, 102, 102, 120], **kw)                 # grande área
    ax.plot([30, 30, 50, 50], [120, 114, 114, 120], **kw)                 # pequena área
    ax.plot([36, 36, 44, 44], [120, 121.5, 121.5, 120], color=TEXT, lw=2, zorder=3)  # gol
    ax.scatter([40], [108], s=6, color=color, zorder=3)                    # marca do pênalti
    t = np.linspace(0, 2 * np.pi, 400)
    ax_x, ax_y = 40 + 10 * np.cos(t), 108 + 10 * np.sin(t)
    m = ax_y < 102
    ax.plot(ax_x[m], ax_y[m], **kw)                                        # meia-lua
    ax.set_xlim(-1, 81); ax.set_ylim(79, 122.5)
    ax.set_aspect("equal"); ax.axis("off")


def xg_maps(grids: dict[str, pd.DataFrame], save=None):
    """Mapas de xG lado a lado (ex.: chute com o pé vs. cabeceio)."""
    n = len(grids)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 4.6), facecolor=BG)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.16, wspace=0.08)
    for ax, (title, g) in zip(np.atleast_1d(axes), grids.items()):
        ax.set_facecolor(BG)
        piv = g.pivot_table(index="x", columns="y", values="xg", dropna=False)
        mesh = ax.pcolormesh(piv.columns.values, piv.index.values, piv.values, cmap="Blues",
                             vmin=0, vmax=0.6, shading="nearest", zorder=1)
        cs = ax.contour(piv.columns.values, piv.index.values, piv.values, levels=[0.05, 0.1, 0.2, 0.4],
                        colors="white", linewidths=0.8, zorder=2)
        ax.clabel(cs, fmt=lambda v: f"{v:.2f}", fontsize=8)
        draw_final_third(ax)
        ax.set_title(title, size=12.5, weight="bold", color=TEXT)
    cax = fig.add_axes([0.3, 0.08, 0.4, 0.035])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cb.set_label("xG (probabilidade de gol)", color=SUB, size=9.5); cb.outline.set_visible(False)
    fig.text(0.02, 0.95, "Onde o chute vale mais: xG previsto pelo modelo", ha="left", size=15, weight="bold", color=TEXT)
    fig.text(0.02, 0.895, "Bola rolando, após passe curto, sem pressão, 1 defensor no caminho, goleiro posicionado. "
             "Áreas com poucos chutes reais ficam em branco.", size=9.5, color=SUB)
    if save:
        fig.savefig(save, dpi=160, bbox_inches="tight", facecolor=BG)
    return fig


def calibration(test: pd.DataFrame, cols: dict[str, str], save=None):
    """Curva de calibração: xG médio previsto vs. taxa real de gols, por faixa."""
    fig, ax = plt.subplots(figsize=(7.5, 6.5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.plot([0, 0.8], [0, 0.8], ls="--", color="#c3c8ce", lw=1, label="Calibração perfeita")
    styles = [dict(color=ACCENT, lw=2.5, marker="o"), dict(color=TEXT, lw=1.5, marker="s", alpha=0.7),
              dict(color=MUTED, lw=1.5, marker="^")]
    for (label, col), st in zip(cols.items(), styles):
        frac, mean = calibration_curve(test["goal"], test[col], n_bins=12, strategy="quantile")
        ax.plot(mean, frac, label=label, ms=5, **st)
    ax.set_xlabel("xG médio previsto", color=TEXT); ax.set_ylabel("Taxa real de gols", color=TEXT)
    ax.set_xlim(0, 0.8); ax.set_ylim(0, 0.8)
    ax.legend(frameon=False, loc="upper left")
    _style(ax, "Calibração: quando o modelo diz 30%, sai gol 30% das vezes?",
           "Chutes do conjunto de teste agrupados em 12 faixas de xG")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=160, bbox_inches="tight", facecolor=BG)
    return fig


def importance(imp: pd.Series, save=None):
    """Importância por permutação (aumento do log loss ao embaralhar a feature)."""
    imp = imp.sort_values()
    fig, ax = plt.subplots(figsize=(8, 6.5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.barh([FEATURE_LABELS.get(i, i) for i in imp.index], imp.values, color=ACCENT, height=0.65)
    ax.set_xlabel("Piora no log loss ao embaralhar a variável", color=TEXT)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e2e2e2", ls="--"); ax.set_axisbelow(True)
    _style(ax, "O que mais pesa no xG", "Importância por permutação no conjunto de teste")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=160, bbox_inches="tight", facecolor=BG)
    return fig


def finishing(players: pd.DataFrame, save=None):
    """Gols − xG por jogador: quem finaliza acima/abaixo do esperado."""
    d = players.sort_values("diff")
    colors = [ACCENT if v > 0 else MUTED for v in d["diff"]]
    fig, ax = plt.subplots(figsize=(8.5, 6.5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.barh(d["nome"], d["diff"], color=colors, height=0.65)
    ax.axvline(0, color=SUB, lw=1)
    for i, (v, c, g, x) in enumerate(zip(d["diff"], d["chutes"], d["gols"], d["xg"])):
        ax.text(v + (1.5 if v > 0 else -1.5), i, f"{int(g)} gols · {x:.0f} xG",
                va="center", ha="left" if v > 0 else "right", size=8.5, color=SUB)
    ax.set_xlabel("Gols − xG (sem pênaltis)", color=TEXT)
    ax.tick_params(axis="y", length=0)
    lim = d["diff"].abs().max() * 1.45
    ax.set_xlim(-lim * 0.45, lim)
    _style(ax, "Finalizadores acima e abaixo do esperado",
           "Jogadores com 150+ chutes no StatsBomb Open Data · xG fora da amostra (validação cruzada)")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=160, bbox_inches="tight", facecolor=BG)
    return fig


SHORT_NAMES = {
    "Lionel Andrés Messi Cuccittini": "Lionel Messi", "Luis Alberto Suárez Díaz": "Luis Suárez",
    "Kylian Mbappé Lottin": "Kylian Mbappé", "Samuel Eto''o Fils": "Samuel Eto'o",
    "Gonzalo Gerardo Higuaín": "Gonzalo Higuaín", "Alexis Alejandro Sánchez Sánchez": "Alexis Sánchez",
    "Martin Braithwaite Christensen": "Martin Braithwaite", "Andrés Iniesta Luján": "Andrés Iniesta",
    "Romelu Lukaku Menama": "Romelu Lukaku", "Gerard Piqué Bernabéu": "Gerard Piqué",
    "Cristiano Ronaldo dos Santos Aveiro": "Cristiano Ronaldo", "Neymar da Silva Santos Junior": "Neymar",
    "Philippe Coutinho Correia": "Philippe Coutinho", "Sergio Busquets i Burgos": "Sergio Busquets",
    "Antoine Griezmann": "Antoine Griezmann", "Karim Benzema": "Karim Benzema",
}


def short_name(n: str) -> str:
    return SHORT_NAMES.get(n, n.replace("''", "'"))
