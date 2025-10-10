# core/entities.py
from dataclasses import dataclass
from typing import Literal
import numpy as np

Sex = Literal["F", "M"]  # тип пола для читаемости

@dataclass(slots=True)
class Organism:
    """
    Особь без логики. Вся динамика (выживание, размножение) реализуется в других модулях.
    Поля-фенотипы:
      - color, lifestyle, activity ∈ [0, 1]
      - speed, aggression, strength ≥ 0
    Служебные:
      - age: целочисленный возраст (тики)
      - injury: степень травмы ∈ [0, 1]
      - alive: флаг «жив/мёртв» (мы удаляем мёртвых из списка, но флаг оставляем для совместимости)
    """
    # --- фенотипы ---
    color: float
    speed: float
    lifestyle: float
    activity: float
    aggression: float
    strength: float
    sex: Sex  # "F" или "M"

    # --- служебные поля ---
    age: int = 0
    injury: float = 0.0
    alive: bool = True

    def __post_init__(self) -> None:
        """Сразу после создания приводим значения к допустимым диапазонам."""
        self.clamp()

    def clamp(self) -> None:
        """
        Жёстко ограничиваем признаки в допустимых диапазонах:
          - [0, 1] для color/lifestyle/activity/injury,
          - [0, +∞) для speed/aggression/strength.
        """
        # признаки [0,1]
        self.color     = float(np.clip(self.color, 0.0, 1.0))
        self.lifestyle = float(np.clip(self.lifestyle, 0.0, 1.0))
        self.activity  = float(np.clip(self.activity, 0.0, 1.0))
        self.injury    = float(np.clip(self.injury, 0.0, 1.0))

        # признаки ≥ 0
        self.speed      = float(max(0.0, self.speed))
        self.aggression = float(max(0.0, self.aggression))
        self.strength   = float(max(0.0, self.strength))
