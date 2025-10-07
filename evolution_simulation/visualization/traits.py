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
    t: int,
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
    df_t = df[df["t"] == t]
    if df_t.empty:
        raise ValueError(f"В таблице нет данных для t={t}. Сначала снимите срез recorder.snapshot(t, ...).")

    # выбираем порядок видов
    if species_order is None:
        species_order = list(df_t["species"].unique())

    plt.figure(figsize=(7.5, 4.5))

    # рисуем по видам
    for sp_name in species_order:
        speeds = df_t.loc[df_t["species"] == sp_name, "speed"].values
        if len(speeds) == 0:
            continue
        # density=False => классическая гистограмма по частотам;
        # alpha — полупрозрачность, чтобы гистограммы было видно поверх друг друга.
        plt.hist(speeds, bins=bins, alpha=0.45, label=sp_name)

    plt.xlabel("speed")
    plt.ylabel("Count")
    plt.title(title if title is not None else f"Распределение скорости по видам (t={t})")
    plt.legend(title="Species")
    plt.tight_layout()
    plt.show()
