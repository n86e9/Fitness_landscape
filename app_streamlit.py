# app_streamlit.py
# Запуск: streamlit run app_streamlit.py

import sys, os
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import numpy as np
import pandas as pd
import streamlit as st
from plotly import graph_objects as go
from plotly.subplots import make_subplots

# --- модули симуляции ---
from evolution_simulation.core.params import SimulationParams
from evolution_simulation.core.entities import Organism
from evolution_simulation.core.simulation import Simulation
from evolution_simulation.core.recorder import TraitRecorder

# Убрали lifestyle и activity
ALL_TRAITS = ["speed", "aggression", "injury", "strength", "color", "age"]

# ================== УТИЛИТЫ СИМУЛЯЦИИ ==================

def init_population(n: int, rng: np.random.Generator) -> list[Organism]:
    """Инициализируем популяцию с правдоподобной вариацией признаков (без lifestyle/activity)."""
    pop = []
    for _ in range(n):
        sex = 'F' if rng.random() < 0.5 else 'M'
        ind = Organism(
            color=float(np.clip(rng.normal(0.5, 0.1), 0, 1)),
            speed=float(max(0.0, rng.normal(1.8, 0.5))),
            aggression=float(max(0.0, rng.normal(0.3, 0.15))),
            strength=float(max(0.0, rng.normal(1.0, 0.3))),
            sex=sex,
            injury=0.0,
            age=0,
            alive=True
        )
        ind.clamp()
        pop.append(ind)
    return pop

@st.cache_data(show_spinner=False)
def run_simulation(total_ticks: int, params: SimulationParams, seed: int = 7, N0: int = 160) -> pd.DataFrame:
    """Запуск симуляции и сбор треков признаков в DataFrame."""
    rng = np.random.default_rng(seed)
    pop = init_population(N0, rng)
    sim = Simulation(population=pop, params=params)
    rec = TraitRecorder()
    for _ in range(total_ticks):
        sim.step()
        rec.snapshot(sim.tick, sim.population)
        if sim.tick % params.season_length == 0:
            rec.snapshot(sim.tick, sim.population)
    return rec.to_dataframe()

# ================== ГРАФИКИ (Plotly) ==================

def make_population_fig(df: pd.DataFrame,
                        show_male: bool,
                        show_female: bool) -> go.Figure:
    """Численность популяции по времени: Total + опционально M/F."""
    if "tick" not in df.columns and "t" in df.columns:
        df = df.rename(columns={"t": "tick"})
    fig = go.Figure()

    # Total
    counts = df.groupby("tick").size().reset_index(name="N")
    fig.add_trace(go.Scatter(x=counts["tick"], y=counts["N"], mode="lines", name="Total N"))

    # By sex (если есть колонка sex)
    if "sex" in df.columns:
        if show_male:
            male = df[df["sex"] == "M"].groupby("tick").size().reindex(counts["tick"], fill_value=0)
            fig.add_trace(go.Scatter(x=counts["tick"], y=male.values, mode="lines", name="Males"))
        if show_female:
            female = df[df["sex"] == "F"].groupby("tick").size().reindex(counts["tick"], fill_value=0)
            fig.add_trace(go.Scatter(x=counts["tick"], y=female.values, mode="lines", name="Females"))

    fig.update_layout(title="Population size over time",
                      xaxis_title="tick", yaxis_title="N", template="plotly_white",
                      legend={"orientation":"h","y":1.02,"yanchor":"bottom"})
    fig.update_xaxes(showgrid=True); fig.update_yaxes(showgrid=True)
    return fig

def make_traits_timeseries_fig(df: pd.DataFrame, traits: list[str], stat: str) -> go.Figure:
    """Временные ряды выбранных признаков (mean/median/std)."""
    if "tick" not in df.columns and "t" in df.columns:
        df = df.rename(columns={"t": "tick"})
    fig = go.Figure()
    agg = {"mean": "mean", "median": "median", "std": "std"}[stat]
    present = [t for t in traits if t in df.columns]
    for tr in present:
        s = df.groupby("tick")[tr].agg(agg)
        fig.add_trace(go.Scatter(x=s.index.values, y=s.values, mode="lines", name=f"{tr} [{stat}]"))
    title = "Traits over time" if present else "Select traits in the sidebar"
    fig.update_layout(title=title, xaxis_title="tick", yaxis_title=stat, template="plotly_white",
                      legend={"orientation":"h","y":1.02,"yanchor":"bottom"})
    fig.update_xaxes(showgrid=True); fig.update_yaxes(showgrid=True)
    return fig

def make_final_histograms(df: pd.DataFrame, traits: list[str], bins: int = 30) -> go.Figure:
    """Гистограммы распределений признаков на последнем тике (в одном ряду)."""
    if "tick" not in df.columns and "t" in df.columns:
        df = df.rename(columns={"t": "tick"})
    if df.empty:
        return go.Figure()

    tmax = int(df["tick"].max())
    dft = df[df["tick"] == tmax]
    traits = [tr for tr in traits if tr in dft.columns]
    cols = max(1, len(traits))

    fig = make_subplots(rows=1, cols=cols, subplot_titles=traits)
    for i, tr in enumerate(traits, start=1):
        fig.add_trace(
            go.Histogram(x=dft[tr].values, nbinsx=bins, name=tr, showlegend=False),
            row=1, col=i
        )
    fig.update_layout(
        title=f"Final-tick histograms (t={tmax})",
        template="plotly_white",
        bargap=0.05,
        height=360
    )
    return fig

# ============== 3D FITNESS: «выживание × гора × выгода скорости» ==============

def selection_score_xy(speed: np.ndarray, aggression: np.ndarray, p: SimulationParams) -> np.ndarray:
    """Гауссов «холм» по (speed, aggression) с максимумом в (opt_speed, opt_aggr)."""
    opt_s = getattr(p, "opt_speed", 1.8)
    opt_a = getattr(p, "opt_aggr", 0.3)
    ss = max(getattr(p, "sel_sigma_speed", 0.6), 1e-6)
    sa = max(getattr(p, "sel_sigma_aggr", 0.4),  1e-6)
    w  = getattr(p, "sel_weight", 1.0)
    ds = (speed - opt_s)
    da = (aggression - opt_a)
    z = (ds*ds)/(2.0*ss*ss) + (da*da)/(2.0*sa*sa)
    return np.exp(- w * z)

def survival_tick(speed: np.ndarray, injury: float, p: SimulationParams,
                  speed_cost_scale: float) -> np.ndarray:
    """Мгновенная выживаемость за тик."""
    speed_term  = (speed_cost_scale * p.speed_surv_cost) * (speed ** 2)
    injury_term = p.injury_survival_penalty * float(injury)
    base = p.surv_base * np.exp(- (speed_term + injury_term))
    return np.clip(base, 0.0, 1.0)

def lambda_effective(speed: np.ndarray, p: SimulationParams, k_speed: float) -> np.ndarray:
    """Насыщающаяся выгода скорости в размножении."""
    sat = speed / (k_speed + speed + 1e-9)
    return p.lambda_base * sat

def make_fitness_surface(df: pd.DataFrame, xtrait: str, ytrait: str,
                         age_const: float, injury_const: float,
                         params: SimulationParams,
                         k_speed: float,
                         speed_cost_scale: float,
                         normalize: bool,
                         xy_span: float,
                         z_scale: float) -> go.Figure:
    """3D-поверхность ландшафта приспособленности с управляемым масштабом по XY и Z."""

    # базовые домены, чтобы не уходить в абсурдные диапазоны
    def trait_domain(tr):
        if tr == "speed":
            return (0.0, 5.0)
        elif tr in ("aggression", "strength"):
            return (0.0, 3.0)
        else:
            return (0.0, 1.0)

    # диапазоны по данным (или дефолт)
    def bounds_for(tr):
        if tr == "speed":
            if tr in df.columns and len(df) > 0:
                q1, q2 = df[tr].quantile([0.05, 0.95]).values
                return (max(0.0, float(q1)), max(float(q2), 0.2))
            return (0.0, 4.0)
        elif tr in ("aggression", "strength"):
            if tr in df.columns and len(df) > 0:
                q1, q2 = df[tr].quantile([0.05, 0.95]).values
                return (max(0.0, float(q1)), max(float(q2), 0.2))
            return (0.0, 3.0)
        else:
            return (0.0, 1.0)

    def expand_range(lo, hi, tr, mult):
        """Расширяем/сжимаем базовый интервал по коэффициенту xy_span."""
        dom_lo, dom_hi = trait_domain(tr)
        base_w = max(hi - lo, 1e-6)
        center = 0.5 * (lo + hi)
        half = 0.5 * base_w * mult
        new_lo = max(dom_lo, center - half)
        new_hi = min(dom_hi, center + half)
        if new_hi - new_lo < 1e-6:
            return dom_lo, dom_hi
        return new_lo, new_hi

    x_lo0, x_hi0 = bounds_for(xtrait)
    y_lo0, y_hi0 = bounds_for(ytrait)
    x_lo, x_hi = expand_range(x_lo0, x_hi0, xtrait, xy_span)
    y_lo, y_hi = expand_range(y_lo0, y_hi0, ytrait, xy_span)

    nx = ny = 80
    X = np.linspace(x_lo, x_hi, nx); Y = np.linspace(y_lo, y_hi, ny)
    XX, YY = np.meshgrid(X, Y)

    speed_median = float(df["speed"].median()) if "speed" in df.columns and len(df) else 1.5
    speed_field = XX if xtrait=="speed" else (YY if ytrait=="speed" else np.full_like(XX, speed_median))
    aggr_field  = XX if xtrait=="aggression" else (YY if ytrait=="aggression" else np.zeros_like(XX))

    S   = survival_tick(speed_field, float(injury_const), params, speed_cost_scale=speed_cost_scale)
    Sel = selection_score_xy(speed_field, aggr_field, params)
    Lam = lambda_effective(speed_field, params, k_speed=k_speed)
    Z   = np.clip(S * Sel * Lam, 0.0, None)

    if normalize:
        zmin = float(np.nanmin(Z)); zmax = float(np.nanmax(Z))
        if zmax > zmin:
            Z = (Z - zmin) / (zmax - zmin + 1e-12)

    Z = Z * z_scale  # уменьшаем высоту холма

    surface = go.Surface(
        x=X, y=Y, z=Z, colorscale="Viridis",
        colorbar={"title": "fitness"},
        contours={"z":{"show":True,"usecolormap":True,"highlightcolor":"lime","project_z":True}}
    )
    fig = go.Figure(data=[surface])
    fig.update_layout(
        title=f"Fitness landscape: {xtrait} vs {ytrait} | injury={injury_const:.2f}",
        scene=dict(
            xaxis_title=xtrait, yaxis_title=ytrait, zaxis_title="fitness",
            xaxis=dict(showgrid=True), yaxis=dict(showgrid=True), zaxis=dict(showgrid=True),
        ),
        template="plotly_white", height=560, margin=dict(l=0, r=0, t=50, b=0)
    )
    return fig

# ================== UI ==================

st.set_page_config(page_title="Evolution Simulation — Streamlit", layout="wide")
st.title("Evolution Simulation — Streamlit Dashboard")

with st.sidebar:
    # базовые контролы
    ticks = st.slider("Ticks to simulate", 20, 3000, 400, step=10)
    run_btn = st.button("Run simulation", use_container_width=True)

    st.markdown("---")
    # Переключатели показа полов на графике численности
    show_male = st.checkbox("Show males on N(t)", value=True)
    show_female = st.checkbox("Show females on N(t)", value=True)

    st.markdown("---")
    traits_selected = st.multiselect("Time-series traits", ALL_TRAITS, default=["speed","aggression","injury"])
    stat = st.radio("Statistic", ["mean", "median", "std"], horizontal=True)

    st.markdown("---")
    st.subheader("3D: base knobs")
    k_speed = st.slider("λ: k_speed (speed saturation)", 0.2, 3.0, 1.0, step=0.1)
    speed_cost_scale = st.slider("Survival speed cost ×", 0.5, 5.0, 2.0, step=0.1)
    normalize = st.checkbox("Normalize fitness to [0,1]", value=True)

    st.markdown("---")
    st.subheader("Axes for 3D")
    col1, col2 = st.columns(2)
    with col1:
        xtrait = st.selectbox("X axis trait", ALL_TRAITS, index=ALL_TRAITS.index("speed"))
    with col2:
        ytrait = st.selectbox("Y axis trait", ALL_TRAITS, index=ALL_TRAITS.index("aggression"))
    age_const = st.slider("Age (for landscape, visual only)", 0, 80, 10, step=1)
    injury_const = st.slider("Injury (for landscape)", 0.0, 1.0, 0.1, step=0.02)

    st.markdown("---")
    st.subheader("Final-tick histograms")
    hist_traits = st.multiselect("Traits for histograms", ALL_TRAITS, default=["speed","aggression","injury"])
    hist_bins = st.slider("Bins", 10, 80, 30, step=5)

# Параметры по умолчанию (согласованы с core/*)
DEFAULT_PARAMS = SimulationParams(
    season_length=6, K=800, seed=123,
    surv_base=0.992, speed_surv_cost=0.003,
    injury_severity_rate=0.01, injury_recovery_rate=0.02,
    injury_survival_penalty=0.25, injury_fertility_penalty=0.40,
    rho_enc_base=1.0/90.0,
    lambda_base=1.25, max_children_per_female=3, dens_alpha=3.0,
    seg_sigma=0.05, mutation=0.05, mutation_sigma=0.04,
    opt_speed=1.8, opt_aggr=0.3, sel_sigma_speed=0.6, sel_sigma_aggr=0.4, sel_weight=1.0
)

# Кэш симуляции: пересчёт только при нажатии кнопки
if run_btn or "df" not in st.session_state:
    st.session_state["df"] = run_simulation(ticks, DEFAULT_PARAMS, seed=DEFAULT_PARAMS.seed)

df = st.session_state["df"]

# ------- Первая строка: N(t) (с опциями M/F) и временные ряды признаков -------
colA, colB = st.columns(2)
with colA:
    st.plotly_chart(make_population_fig(df, show_male=show_male, show_female=show_female),
                    use_container_width=True)
with colB:
    st.plotly_chart(make_traits_timeseries_fig(df, traits_selected, stat), use_container_width=True)

# ------- Вторая строка: слева гистограммы, справа 3D-ландшафт + инлайн-рычаги масштаба -------
colL, colR = st.columns([1, 2], gap="large")
with colL:
    st.plotly_chart(make_final_histograms(df, hist_traits, hist_bins), use_container_width=True)
with colR:
    st.subheader("Landscape scale")
    c1, c2, c3 = st.columns(3)
    xy_span = c1.slider("XY span ×", 1.0, 5.0, 3.0, step=0.5, key="xy_span_inline",
                        help="Расширяет охват по осям X/Y — холм выглядит меньше.")
    z_scale = c2.slider("Z scale ×", 0.1, 1.0, 0.5, step=0.05, key="z_scale_inline",
                        help="Масштаб высоты по оси Z — уменьшает высоту холма.")
    normalize_inline = c3.checkbox("Normalize [0,1]", value=normalize, key="normalize_inline",
                                   help="Нормировать поверхность по Z в [0,1] перед масштабированием.")
    st.plotly_chart(
        make_fitness_surface(
            df, xtrait, ytrait, age_const, injury_const, DEFAULT_PARAMS,
            k_speed=k_speed, speed_cost_scale=speed_cost_scale, normalize=normalize_inline,
            xy_span=xy_span, z_scale=z_scale
        ),
        use_container_width=True
    )
