# visualization/traits.py
"""
Гибкая визуализация признаков из TraitRecorder-таблицы.

ОСНОВНЫЕ ПАТТЕРНЫ:
- В РЯДЕ ВРЕМЕНИ: plot_trait_timeseries(df, traits=["speed","aggression"], stat="mean")
- ГИСТОГРАММА В МОМЕНТ: plot_hist_at_tick(df, trait="injury", tick=30, bins=40)
- СВЯЗЬ ДВУХ ПРИЗНАКОВ: plot_scatter(df, x="speed", y="strength", tick=None, hue="sex")

Интерактив (необязательно):
- make_interactive_trait_timeseries(df, traits=["speed","aggression","injury"], stat="mean")
  Откроет окно с радиокнопками — можно переключать, какой признак показывать.
"""

from typing import List, Optional, Sequence, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons


# --------- ВСПОМОГАТЕЛЬНЫЕ ШТУКИ ---------

# допустимые признаки-числа (ожидаемые колонки)
NUMERIC_TRAITS = [
    "speed", "aggression", "injury", "strength",
    "color", "lifestyle", "activity", "age",
]

def _require_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    """Проверка наличия колонок с понятной ошибкой."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"В DataFrame нет колонок: {missing}. Доступные: {list(df.columns)}")

def _ensure_tick_column(df: pd.DataFrame) -> None:
    """Гарантируем, что есть колонка времени tick (а не t)."""
    if "tick" not in df.columns and "t" in df.columns:
        df.rename(columns={"t": "tick"}, inplace=True)


# --------- 1) ЧИСЛЕННОСТЬ ПО ВРЕМЕНИ ---------

def plot_population_timeseries(df: pd.DataFrame, by_sex: bool = False):
    """
    Линия N(t). Если by_sex=True — рисуем по полу и суммарно.
    """
    _ensure_tick_column(df)
    _require_columns(df, ["tick"])
    fig, ax = plt.subplots(figsize=(8, 4.5))

    if by_sex and "sex" in df.columns:
        counts = df.groupby(["tick", "sex"]).size().reset_index(name="N")
        pivot = counts.pivot(index="tick", columns="sex", values="N").fillna(0.0)
        for col in pivot.columns:
            ax.plot(pivot.index, pivot[col], label=f"sex={col}")
        ax.plot(pivot.index, pivot.sum(axis=1), linestyle="--", label="Total")
        ax.set_title("Population by sex over time")
    else:
        counts = df.groupby("tick").size().reset_index(name="N")
        ax.plot(counts["tick"], counts["N"], label="Total")
        ax.set_title("Population size over time")

    ax.set_xlabel("tick"); ax.set_ylabel("N")
    ax.legend(); fig.tight_layout()
    return fig, ax


# --------- 2) ГИБКИЙ РЯД ВРЕМЕНИ ДЛЯ ЛЮБЫХ ПРИЗНАКОВ ---------

def plot_trait_timeseries(
    df: pd.DataFrame,
    traits: Sequence[str],
    stat: Literal["mean", "median", "std"] = "mean",
    group_by: Optional[str] = None,
):
    """
    Рисует во времени любую агрегированную статистику по выбранным признакам.
    - traits: список колонок-признаков, напр. ["speed","aggression","injury"]
    - stat: "mean", "median", или "std"
    - group_by: опциональная категориальная колонка для разбиения (например, "sex")
      Если задано, то на каждый trait будет по ЛИНИИ на каждую группу.

    Примеры:
      plot_trait_timeseries(df, ["speed","injury"], stat="mean")
      plot_trait_timeseries(df, ["speed"], stat="median", group_by="sex")
    """
    _ensure_tick_column(df)
    _require_columns(df, ["tick"])
    for tr in traits:
        _require_columns(df, [tr])

    fig, ax = plt.subplots(figsize=(9, 4.5))

    aggfunc = {"mean": "mean", "median": "median", "std": "std"}[stat]

    if group_by is not None and group_by in df.columns:
        for tr in traits:
            g = df.groupby(["tick", group_by])[tr].agg(aggfunc).reset_index()
            for grp_val, sub in g.groupby(group_by):
                ax.plot(sub["tick"], sub[tr], label=f"{tr} ({group_by}={grp_val})")
    else:
        for tr in traits:
            series = df.groupby("tick")[tr].agg(aggfunc)
            ax.plot(series.index, series.values, label=f"{tr} [{stat}]")

    ax.set_xlabel("tick"); ax.set_ylabel(stat)
    ax.set_title(f"{stat} of traits over time")
    ax.legend(ncols=2); fig.tight_layout()
    return fig, ax


# --------- 3) ГИСТОГРАММА ЛЮБОГО ПРИЗНАКА В КОНКРЕТНЫЙ МОМЕНТ ---------

def plot_hist_at_tick(df: pd.DataFrame, trait: str, tick: int, bins: int = 30):
    """
    Гистограмма распределения выбранного признака в момент времени tick.
    """
    _ensure_tick_column(df)
    _require_columns(df, ["tick", trait])

    df_t = df[df["tick"] == tick]
    if df_t.empty:
        raise ValueError(f"Нет данных для tick={tick}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(df_t[trait].to_numpy(), bins=bins, alpha=0.8)
    ax.set_xlabel(trait); ax.set_ylabel("count"); ax.set_title(f"{trait} at tick={tick}")
    fig.tight_layout()
    return fig, ax


# --------- 4) СВЯЗЬ ДВУХ ПРИЗНАКОВ (SCATTER) ---------

def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    tick: Optional[int] = None,
    hue: Optional[str] = None,
    sample: Optional[int] = None,
):
    """
    Точечный график связь x vs y.
    - tick: если задан, берём только этот момент; иначе — все тики (можно подсэмплировать).
    - hue: цветовая категоризация (например, "sex").
    - sample: взять случайную подвыборку N строк (ускоряет и снижает «слипание» точек).

    Примеры:
      plot_scatter(df, "speed", "strength", tick=40, hue="sex")
      plot_scatter(df, "color", "aggression", sample=500)
    """
    _ensure_tick_column(df)
    _require_columns(df, [x, y])

    sub = df if tick is None else df[df["tick"] == tick]
    if sample is not None and len(sub) > sample:
        sub = sub.sample(sample, random_state=0)

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    if hue is not None and hue in sub.columns:
        for val, part in sub.groupby(hue):
            ax.scatter(part[x], part[y], s=12, alpha=0.6, label=f"{hue}={val}")
        ax.legend(loc="best")
    else:
        ax.scatter(sub[x], sub[y], s=12, alpha=0.7)

    title = f"{x} vs {y}" + (f"  (tick={tick})" if tick is not None else "")
    ax.set_title(title)
    ax.set_xlabel(x); ax.set_ylabel(y)
    fig.tight_layout()
    return fig, ax


# --------- 5) ЛЁГКАЯ ИНТЕРАКТИВКА (ОПЦИОНАЛЬНО) ---------

def make_interactive_trait_timeseries(
    df: pd.DataFrame,
    traits: Sequence[str] = ("speed", "aggression", "injury"),
    stat: Literal["mean", "median", "std"] = "mean",
):
    """
    Простая интерактивная панель: слева радио-кнопки со списком traits,
    справа — график выбранного признака по времени (stat из mean/median/std).
    Никаких внешних зависимостей, только matplotlib.widgets.

    Использование:
      fig, ax, ui = make_interactive_trait_timeseries(df, traits=["speed","injury","strength"], stat="mean")
      plt.show()

    Возвращает: (fig, ax, radio_widget)
    """
    _ensure_tick_column(df)
    # предварительно готовим агрегации для всех traits (чтобы переключалось мгновенно)
    data_per_trait = {}
    for tr in traits:
        _require_columns(df, ["tick", tr])
        series = df.groupby("tick")[tr].agg(stat)
        data_per_trait[tr] = (series.index.to_numpy(), series.values)

    # создаём фигуру и оси
    fig = plt.figure(figsize=(9.5, 4.8))
    # область для радио: [left, bottom, width, height] в относительных координатах фигуры
    ax_radio = plt.axes([0.02, 0.15, 0.16, 0.7])
    ax_plot  = plt.axes([0.24, 0.12, 0.74, 0.82])

    # рисуем первый признак
    first = traits[0]
    (x0, y0) = data_per_trait[first]
    (line,) = ax_plot.plot(x0, y0, label=f"{first} [{stat}]")
    ax_plot.set_xlabel("tick")
    ax_plot.set_ylabel(stat)
    ax_plot.set_title("Trait over time")
    ax_plot.legend(loc="best")
    fig.tight_layout()

    # радио-кнопки
    radio = RadioButtons(ax_radio, labels=tuple(traits), active=0)

    def _on_pick(label: str):
        x, y = data_per_trait[label]
        line.set_data(x, y)
        ax_plot.relim()
        ax_plot.autoscale_view()
        # обновляем легенду и заголовок
        ax_plot.legend([f"{label} [{stat}]"])
        ax_plot.set_title(f"Trait over time: {label} ({stat})")
        fig.canvas.draw_idle()

    radio.on_clicked(_on_pick)

    return fig, ax_plot, radio
