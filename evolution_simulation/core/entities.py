from dataclasses import dataclass
from typing import Literal
import numpy as np

@dataclass
class Organism:
    """
    Особь. Никакой логики — только данные; вся динамика живёт в других модулях.
    """
    color: float        # [0,1]
    speed: float        # >=0
    aggression: float   # >=0
    strength: float     # >=0
    sex: Literal["F", "M"]

    # служебные поля
    age: int = 0
    injury: float = 0.0  # [0,1]
    alive: bool = True

    def clamp(self) -> None:
        """Обрезаем значения признаков в допустимые диапазоны."""
        self.color      = float(np.clip(self.color, 0.0, 1.0))
        self.speed      = float(max(0.0, self.speed))
        self.aggression = float(max(0.0, self.aggression))
        self.injury     = float(np.clip(self.injury, 0.0, 1.0))
        self.strength   = float(max(0.0, self.strength))
