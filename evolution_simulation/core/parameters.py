# core/params.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class SimulationParameters:
    """
    Глобальные параметры. Минимальный набор:
    - season_length: через сколько тиков идёт размножение (длина сезона),
    - s_base: базовая выживаемость за тик (чтобы что-то умирали/старели),
    - speed_survival_cost: простая цена скорости (штраф к выживанию),
    - lambda_base: среднее число детёнышей у самки за сезон,
    - мутации: редкие небольшие сдвиги признаков.
    """
    season_length: int = 20
    K: float = 500.0
    seed: Optional[int] = 42

    # простая смертность
    s_base: float = 0.98
    speed_survival_cost: float = 0.01
    injury_rate: float = 0.02

    # размножение
    lambda_base: float = 1.2
    max_children_per_female: int = 4 # макс. число детей у самки за сезон

    # наследование/мутации
    seg_sigma: float = 0.05 # стандартное отклонение при наследовании (сегрегация)
    mutaion: float = 0.05 # вероятность мутации
    mutation_sigma: float = 0.04 # стандартное отклонение при появлении мутации
