# main.py — интерактивная панель (одно окно, 2 графика, сетка, без правого окна)
# ------------------------------------------------------------------------------

# 0) Настройка интерактивного бэкенда ДО pyplot
import matplotlib
for backend in ("QtAgg", "TkAgg", "MacOSX"):
    try:
        matplotlib.use(backend)
        break
    except Exception:
        continue

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons, RadioButtons

# --- твои модули ---
from evolution_simulation.core.params import SimulationParams
from evolution_simulation.core.entities import Organism
from evolution_simulation.core.simulation import Simulation
from evolution_simulation.core.recorder import TraitRecorder

ALL_TRAITS = ["speed", "aggression", "injury", "strength", "color", "lifestyle", "activity", "age"]


# ------------------------ инициализация популяции ------------------------
def init_population(n: int, rng: np.random.Generator) -> list[Organism]:
    pop = []
    for _ in range(n):
        sex = 'F' if rng.random() < 0.5 else 'M'
        ind = Organism(
            color=float(np.clip(rng.normal(0.5, 0.1), 0, 1)),
            speed=float(max(0.0, rng.normal(1.8, 0.5))),
            lifestyle=float(np.clip(rng.normal(0.5, 0.1), 0, 1)),
            activity=float(np.clip(rng.normal(0.7, 0.15), 0, 1)),
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


def run_simulation(total_ticks: int, params: SimulationParams, seed: int = 7, N0: int = 160):
    rng = np.random.default_rng(seed)
    population = init_population(N0, rng)
    sim = Simulation(population=population, params=params)
    rec = TraitRecorder()
    for _ in range(total_ticks):
        sim.step()
        rec.snapshot(sim.tick, sim.population)
        if sim.tick % params.season_length == 0:
            rec.snapshot(sim.tick, sim.population)
    df = rec.to_dataframe()
    return df, sim


# ------------------------ сборка дашборда ------------------------
def build_dashboard():
    # Параметры по умолчанию (подкрутил к более стабильной динамике)
    params = SimulationParams(
        season_length=5,           # чаще сезоны -> меньше «скачков»
        K=800, seed=123,

        surv_base=0.995,           # базовая выживаемость повыше
        speed_surv_cost=0.003,     # цена скорости ниже (раньше была слишком жесткой)

        injury_severity_rate=0.01, # травма копится медленнее
        injury_recovery_rate=0.02, # и заживает быстрее
        injury_survival_penalty=0.25,
        injury_fertility_penalty=0.4,
        rho_enc_base=1.0/80.0,     # реже «встречи» на старте

        encounter_density_weight=0.5,
        injury_step_cap=0.03,

        lambda_base=1.2,           # немного ↑ рождаемость
        max_children_per_female=3,
        dens_alpha=3.5,

        seg_sigma=0.05, mutation=0.05, mutation_sigma=0.04,

        # мягкое старение (см. блок выше)
        age_mature=10,
        age_gompertz_a=0.0005,
        age_gompertz_b=0.08,
        age_hard_cap=120,

        # сексуальный отбор (как было)
        beta_color=0.8, beta_strength=1.0, beta_speed=0.6,
        beta_injury=1.0, beta_overaggr=0.5, aggr_star=0.8,
        beta_lifestyle_in_males=0.0,

        # половые коэффициенты травмы + lifestyle (можно оставить как было)
        male_aggression_injury_factor=0.8,
        female_aggression_injury_factor=0.3,
        lifestyle_conflict_damp=0.5,
        lifestyle_survival_bonus=0.04,
    )

    T0 = 1000  # стартовое число тиков
    df, sim = run_simulation(T0, params)

    present_traits = [t for t in ALL_TRAITS if t in df.columns]
    if "injury" not in present_traits:
        present_traits.insert(0, "speed")  # безопасный дефолт

    # --------- фигура и сетка (два графика справа, панель виджетов слева) ---------
    fig = plt.figure(figsize=(12.5, 6.8))
    fig.canvas.manager.set_window_title("Evolution Simulation — Interactive Dashboard")

    # Правая часть: 2 строки графиков
    gs = fig.add_gridspec(2, 1, left=0.30, right=0.98, top=0.96, bottom=0.10, hspace=0.30)
    axN      = fig.add_subplot(gs[0, 0])  # численность
    axSeries = fig.add_subplot(gs[1, 0])  # временные ряды

    # Левая колонка: панель виджетов
    ax_sTicks   = plt.axes([0.05, 0.84, 0.22, 0.035], facecolor="#f6f6f6")
    ax_rbStat   = plt.axes([0.05, 0.68, 0.22, 0.12],  facecolor="#f6f6f6")
    ax_cbTraits = plt.axes([0.05, 0.32, 0.22, 0.33],  facecolor="#f6f6f6")
    ax_btnUpdate= plt.axes([0.05, 0.18, 0.22, 0.06])

    # --------- виджеты (все ссылки сохраняем) ---------
    sTicks = Slider(ax_sTicks, "Ticks", valmin=50, valmax=2000, valinit=T0, valstep=10)
    rbStat = RadioButtons(ax_rbStat, labels=("mean", "median", "std"), active=0)
    default_checked = set(["speed", "aggression", "injury"])
    labels = tuple(present_traits)
    actives = [lab in default_checked for lab in labels]
    cbTraits = CheckButtons(ax_cbTraits, labels=labels, actives=actives)
    btnUpdate = Button(ax_btnUpdate, "Update / Re-run", color="#e0e0ff", hovercolor="#d0d0ff")

    # Подправим шрифты и сетку
    for a in (axN, axSeries):
        a.title.set_fontsize(10)
        a.xaxis.label.set_fontsize(9)
        a.yaxis.label.set_fontsize(9)
        for ticklbl in (a.get_xticklabels() + a.get_yticklabels()):
            ticklbl.set_fontsize(8)
        a.grid(True, which="both", alpha=0.25)  # ← СЕТКА

    # --------- функции отрисовки ---------
    def draw_population(ax, df_local):
        ax.clear()
        counts = df_local.groupby("tick").size().reset_index(name="N")
        if counts.empty:
            ax.set_title("Population over time (no data)")
            ax.grid(True, alpha=0.25)
            return
        ax.plot(counts["tick"], counts["N"], lw=1.6, label="Total N")
        ax.set_xlabel("tick"); ax.set_ylabel("N")
        ax.set_title("Population size over time")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, which="both", alpha=0.25)

    def draw_series(ax, df_local):
        ax.clear()
        stat = rbStat.value_selected
        agg = {"mean": "mean", "median": "median", "std": "std"}[stat]
        selected = [lab for lab, on in zip(labels, cbTraits.get_status()) if on and lab in df_local.columns]
        if not selected:
            ax.set_title("Select traits on the left")
            ax.set_xlabel("tick"); ax.set_ylabel(stat)
            ax.grid(True, alpha=0.25)
            return
        for tr in selected:
            s = df_local.groupby("tick")[tr].agg(agg)
            ax.plot(s.index.values, s.values, lw=1.4, label=f"{tr} [{stat}]")
        ax.set_xlabel("tick"); ax.set_ylabel(stat)
        ax.set_title("Traits over time")
        ax.legend(ncols=2, fontsize=8)
        ax.grid(True, which="both", alpha=0.25)

    # --------- начальная отрисовка ---------
    draw_population(axN, df)
    draw_series(axSeries, df)
    fig.canvas.draw_idle()

    # --------- обработчики событий ---------
    def on_series_config_change(_):
        # только перерисовка по текущему df (без пересчёта симуляции)
        draw_series(axSeries, df)
        fig.canvas.draw_idle()

    def on_update_clicked(event):
        nonlocal df, sim
        total = int(sTicks.val)
        # Перепрогон симуляции по нажатию кнопки (чтобы не зависало при перетаскивании ползунка)
        df, sim = run_simulation(total, params, seed=params.seed)
        draw_population(axN, df)
        draw_series(axSeries, df)
        fig.canvas.draw_idle()

    rbStat.on_clicked(on_series_config_change)
    cbTraits.on_clicked(on_series_config_change)
    btnUpdate.on_clicked(on_update_clicked)
    # ВАЖНО: не подписываемся на sTicks.on_changed, чтобы он не ломал интерактив при перетаскивании.

    return fig


if __name__ == "__main__":
    fig = build_dashboard()
    plt.show()
