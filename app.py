"""App do modelo de xG: clique no campo e veja a probabilidade de gol do chute.

Rodar localmente:  streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.model import FEATURES, boosting_model, load_shots, shot_scenario
from src.viz import ACCENT, short_name

st.set_page_config(page_title="Modelo de xG", page_icon="⚽", layout="wide")
ROOT = Path(__file__).parent


@st.cache_resource(show_spinner="Treinando o modelo de xG (só na primeira vez)...")
def get_model():
    df = load_shots(ROOT / "data" / "processed" / "shots.csv")
    model = boosting_model().fit(df[FEATURES], df["goal"])
    return model, df


@st.cache_data
def density_grid(_df, step=1.0):
    xs, ys = np.arange(80, 120, step) + step / 2, np.arange(0, 80, step) + step / 2
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    H, _, _ = np.histogram2d(_df["x"], _df["y"], bins=[np.arange(80, 120.01, step), np.arange(0, 80.01, step)])
    # suaviza a contagem num raio de ~2 jardas e usa a simetria lateral
    from scipy.ndimage import uniform_filter
    H = uniform_filter(H + H[:, ::-1], size=5, mode="constant") * 25 / 2
    return pd.DataFrame({"x": X.ravel(), "y": Y.ravel(), "n": H.ravel()})


model, shots = get_model()
dens = density_grid(shots)

st.title("⚽ Modelo de xG")
st.caption("Probabilidade de um chute virar gol · gradient boosting treinado com ~100 mil chutes do "
           "[StatsBomb Open Data](https://github.com/statsbomb/open-data) · "
           "[código no GitHub](https://github.com/andrecasemiro28/modelo-xg)")

tab_sim, tab_perf, tab_fin = st.tabs(["🎯 Simulador de chute", "📏 Desempenho do modelo", "🥅 Finalizadores"])

# ------------------------------------------------------------------ simulador
with tab_sim:
    if "loc" not in st.session_state:
        st.session_state["loc"] = (106.5, 33.5)

    left, right = st.columns([3, 2], gap="large")
    with right:
        st.subheader("Contexto do lance")
        body = st.radio("Parte do corpo", ["Pé", "Cabeça"], horizontal=True)
        c1, c2 = st.columns(2)
        situacao = c1.selectbox("Situação", ["Bola rolando", "Falta direta"])
        assist = c2.selectbox("Assistência", ["Passe curto", "Enfiada", "Cruzamento", "Cruzamento rasteiro",
                                             "Passe alto", "Sem assistência"])
        origem = c1.selectbox("Origem da jogada", ["Jogada normal", "Contra-ataque", "Escanteio",
                                                   "Falta", "Lateral"])
        tecnica = c2.selectbox("Técnica", ["Normal", "Voleio", "Meio-voleio", "Cobertura"])
        n_def = st.slider("Defensores entre a bola e o gol", 0, 5, 1)
        marc = st.slider("Distância do marcador mais próximo (m)", 0.5, 10.0, 2.0, 0.5)
        gk = st.slider("Goleiro adiantado (m da linha do gol)", 0.5, 12.0, 2.5, 0.5)
        o1, o2, o3 = st.columns(3)
        pressao = o1.checkbox("Sob pressão")
        primeira = o2.checkbox("De primeira")
        mano = o3.checkbox("Mano a mano")

    maps = {
        "body_part": {"Pé": "Foot", "Cabeça": "Head"}[body],
        "shot_type": {"Bola rolando": "Open Play", "Falta direta": "Free Kick"}[situacao],
        "assist": {"Passe curto": "passe_curto", "Enfiada": "enfiada", "Cruzamento": "cruzamento",
                   "Cruzamento rasteiro": "cruzamento_rasteiro", "Passe alto": "passe_alto",
                   "Sem assistência": "sem_assistencia"}[assist],
        "play_pattern": {"Jogada normal": "Regular Play", "Contra-ataque": "From Counter",
                         "Escanteio": "From Corner", "Falta": "From Free Kick", "Lateral": "From Throw In"}[origem],
        "technique": {"Normal": "Normal", "Voleio": "Volley", "Meio-voleio": "Half Volley", "Cobertura": "Lob"}[tecnica],
        "under_pressure": int(pressao), "first_time": int(primeira), "one_on_one": int(mano),
        "n_def_in_cone": n_def, "nearest_opp": marc * 1.094, "gk_off_line": gk * 1.094,
        "n_opp_5m": max(n_def, int(marc <= 5)),
    }
    if maps["shot_type"] == "Free Kick":
        maps.update(assist="sem_assistencia", play_pattern="From Free Kick")

    grid = pd.DataFrame([shot_scenario(x, y, **maps) for x, y in zip(dens["x"], dens["y"])])
    dens["xg"] = model.predict_proba(grid[FEATURES])[:, 1]
    sparse = dens["n"] < 15

    with left:
        st.markdown("**Clique em qualquer ponto do campo** para posicionar o chute.")
        fig = go.Figure()
        rare = dens[sparse]
        fig.add_trace(go.Scatter(
            x=rare["y"], y=rare["x"], mode="markers", name="",
            marker=dict(symbol="square", size=9, color="#ebe9e4", line=dict(width=0)),
            customdata=np.c_[rare["xg"], rare["n"]],
            hovertemplate="xG %{customdata[0]:.2f} (poucos dados)<extra></extra>",
        ))
        dense = dens[~sparse]
        fig.add_trace(go.Scatter(
            x=dense["y"], y=dense["x"], mode="markers",
            marker=dict(symbol="square", size=9, color=dense["xg"], colorscale="Blues",
                        cmin=0, cmax=0.6, line=dict(width=0),
                        colorbar=dict(title="xG", thickness=12, len=0.6)),
            customdata=np.c_[dense["xg"], dense["n"]],
            hovertemplate="xG %{customdata[0]:.2f}<extra></extra>", name="",
        ))
        lx, ly = st.session_state["loc"]
        fig.add_trace(go.Scatter(x=[ly], y=[lx], mode="markers", hoverinfo="skip",
                                 marker=dict(size=18, color="#ffffff", line=dict(color="#1f2328", width=3))))
        line = dict(color="#8b949e", width=1.5)
        for x0, y0, x1, y1 in [(0, 80, 80, 120), (18, 102, 62, 120), (30, 114, 50, 120)]:
            fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, line=line)
        fig.add_shape(type="rect", x0=36, y0=120, x1=44, y1=121.5, line=dict(color="#1f2328", width=3))
        t = np.linspace(0, 2 * np.pi, 200); ax_, ay_ = 40 + 10 * np.cos(t), 108 + 10 * np.sin(t)
        fig.add_trace(go.Scatter(x=ax_[ay_ < 102], y=ay_[ay_ < 102], mode="lines", line=line, hoverinfo="skip"))
        fig.update_layout(height=430, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
                          plot_bgcolor="#f8f7f4", paper_bgcolor="rgba(0,0,0,0)", clickmode="event+select",
                          xaxis=dict(range=[-1, 81], visible=False, fixedrange=True),
                          yaxis=dict(range=[79, 122.5], visible=False, scaleanchor="x", fixedrange=True))
        ev = st.plotly_chart(fig, on_select="rerun", selection_mode="points", key="campo", width="stretch")
        if ev and ev.selection and ev.selection.points:
            p = ev.selection.points[0]
            new = (p["y"], p["x"])
            if new != st.session_state["loc"]:
                st.session_state["loc"] = new
                st.rerun()

    with right:
        x, y = st.session_state["loc"]
        row = pd.DataFrame([shot_scenario(x, y, **maps)])
        xg = float(model.predict_proba(row[FEATURES])[:, 1][0])
        dist_m = row["distance"].iloc[0] * 0.9144
        st.markdown("---")
        m1, m2 = st.columns(2)
        m1.metric("xG do chute", f"{xg:.2f}")
        m2.metric("Distância ao gol", f"{dist_m:.0f} m")
        st.markdown(f"Esse chute vira gol em **{xg:.0%}** das vezes, cerca de **1 a cada {max(1, round(1 / xg))}** chutes.")
        n_here = int(dens.loc[((dens["x"] - x).abs() < 0.6) & ((dens["y"] - y).abs() < 0.6), "n"].max())
        if n_here < 15:
            st.warning("Há poucos chutes reais nessa região do campo, então o modelo está extrapolando.")

# ------------------------------------------------------------------ desempenho
with tab_perf:
    st.markdown("Validação cruzada com 5 folds **agrupados por partida**, comparada com o xG oficial da StatsBomb.")
    cv = pd.read_csv(ROOT / "reports" / "metricas_cv.csv", index_col=0)
    st.dataframe(cv.style.format("{:.4f}"), width="stretch")
    st.caption("Log loss e Brier: menor é melhor. ROC AUC: maior é melhor.")
    c1, c2 = st.columns(2)
    c1.image(str(ROOT / "reports" / "figures" / "calibracao.png"))
    c2.image(str(ROOT / "reports" / "figures" / "importancia.png"))
    st.image(str(ROOT / "reports" / "figures" / "mapa_xg.png"))

# ------------------------------------------------------------------ finalizadores
with tab_fin:
    st.markdown("Gols sem pênalti **menos** xG previsto fora da amostra. Valores positivos indicam um jogador "
                "que finaliza **acima do esperado**.")
    with st.expander("Habilidade ou sorte? Como ler a coluna z"):
        st.markdown(
            "Cada chute é tratado como uma moeda que vira gol com probabilidade *p* (o xG). Se o jogador finalizasse "
            "como a média, marcaria em média **Σp** gols, com variância **Σp(1−p)**. "
            "O **z = (gols − xG) / desvio padrão** mostra quantos desvios ele está acima do esperado: "
            "com **|z| > 2**, a diferença dificilmente é sorte.")
    fin = pd.read_csv(ROOT / "data" / "processed" / "finalizadores.csv")
    fin["player"] = fin["player"].map(short_name)
    minimo = st.slider("Mínimo de chutes", 50, 500, 150, 25)
    f = fin[fin["chutes"] >= minimo].sort_values("diff", ascending=False)
    so_sig = st.checkbox("Mostrar só diferenças significativas (|z| > 2)")
    if so_sig:
        f = f[f["z"].abs() > 2]
    f = f[["player", "chutes", "gols", "xg", "diff", "desvio_padrao", "z"]]
    st.dataframe(
        f.rename(columns={"player": "Jogador", "chutes": "Chutes", "gols": "Gols", "xg": "xG", "diff": "Gols − xG",
                          "desvio_padrao": "Desvio padrão", "z": "z (desvios)"}),
        hide_index=True, width="stretch",
        column_config={"Gols − xG": st.column_config.NumberColumn(format="%+.1f"),
                       "xG": st.column_config.NumberColumn(format="%.1f"),
                       "Desvio padrão": st.column_config.NumberColumn(format="%.1f"),
                       "z (desvios)": st.column_config.NumberColumn(format="%+.1f")})
    st.caption("Somente futebol masculino. O Open Data tem muitos jogos do Barcelona, o que favorece jogadores daquele elenco.")
