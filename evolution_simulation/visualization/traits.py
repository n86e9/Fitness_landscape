# viz/traits.py
"""
Визуализация вариации признаков.
Здесь — одна функция: гистограммы скорости (speed) по видам в ОДИН выбранный момент времени.
Без наворотов: одна картинка, виды накладываются полупрозрачно.
"""

import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, List, Dict

def plot_speed_hist_at_time(
    df: pd.DataFrame,
    tick: int,
    species_order: Optional[List[str]] = None,
    bins: int = 30,
    title: Optional[str] = None,
) -> None:
    """
    Нарисовать гистограмму распределения 'speed' по ВИДАМ на шаге времени t.
    - df: таблица из recorder.to_dataframe()
    - t: интересующий момент времени
    - species_order: порядок отображения легенды (по умолчанию — как в данных)
    - bins: число бинов гистограммы
    - title: заголовок графика (если None — поставим авто)
    
    Принципы:
    - Полупрозрачные гистограммы разных видов накладываются на одни оси (сравнение форм и позиций),
    - Оси и легенда подписаны, цветовую схему выбирает matplotlib автоматически.
    """
    # фильтруем по моменту времени
    df_tick = df[df["tick"] == tick]
    if df_tick.empty:
        raise ValueError(f"В таблице нет данных для t={tick}. Сначала снимите срез recorder.snapshot(t, ...).")

    # выбираем порядок видов
    if species_order is None:
        species_order = list(df_tick["species"].unique())

    plt.figure(figsize=(7.5, 4.5))

    # рисуем по видам
    for specie_name in species_order:
        speeds = df_tick.loc[df_tick["species"] == specie_name, "speed"].values
        if len(speeds) == 0:
            continue
        # density=False => классическая гистограмма по частотам;
        # alpha — полупрозрачность, чтобы гистограммы было видно поверх друг друга.
        plt.hist(speeds, bins=bins, alpha=0.45, label=specie_name)

    plt.xlabel("speed")
    plt.ylabel("Count")
    plt.title(title if title is not None else f"Распределение скорости по видам (t={tick})")
    plt.legend(title="Species")
    plt.tight_layout()
    plt.show()
