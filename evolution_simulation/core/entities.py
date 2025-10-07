# core/entities.py
from dataclasses import dataclass, field
from typing import List, Literal

Sex = Literal["F", "M"] # возможные значения пола

@dataclass
class Organism:
    """
    Особь без логики. Все вычисления будут в отдельных модулях.
    """
    color: float       # [0,1]
    speed: float       # >=0
    lifestyle: float   # [0,1]
    activity: float    # [0,1]
    aggression: float  # >=0
    strength: float    # >=0
    sex: Sex
    age: int = 0
    alive: bool = True
    injury: bool = False

    def clamp(self) -> None: #
        """Ограничиваем параметры занчений в диапазоне от 0 до 1."""
        self.color = float(min(1.0, max(0.0, self.color)))
        self.lifestyle = float(min(1.0, max(0.0, self.lifestyle)))
        self.activity = float(min(1.0, max(0.0, self.activity)))
        self.speed = float(max(0.0, self.speed))
        self.aggression = float(max(0.0, self.aggression))
        self.strength = float(max(0.0, self.strength))

@dataclass
class Species:
    """
    Просто контейнер для группы особей + имя.
    Даже без изоляции полезно:
      - считать статистику и строить графики по видам,
      - задавать разные начальные распределения,
      - в будущем ограничить скрещивание между видами.
    """
    name: str
    individuals: List[Organism] = field(default_factory=list)

    def alive_count(self) -> int:
        return sum(1 for x in self.individuals if x.alive)
