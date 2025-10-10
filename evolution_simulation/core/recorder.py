"""
TraitRecorder — сборщик срезов признаков во времени.
Снимаем срез после каждого tика (и/или после сезона), потом строим графики.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import pandas as pd
from .entities import Organism

@dataclass
class TraitRecorder:
    # сырые записи (каждая строка — dict)
    rows: List[Dict[str, Any]] = field(default_factory=list)

    def snapshot(self, tick: int, population: List[Organism]) -> None:
        """
        Снять срез на момент времени tick.
        Для каждой ЖИВОЙ особи сохраняем пол, возраст и ключевые признаки.
        """
        for ind in population:
            # мы уже удаляем мёртвых в Simulation, но на всякий случай:
            if not ind.alive:
                continue

            # фиксируем признаки особи в строку
            self.rows.append({
                "tick": tick,
                "sex": ind.sex,
                "age": ind.age,
                "color": ind.color,
                "speed": ind.speed,
                "aggression": ind.aggression,
                "strength": ind.strength,
                "injury": float(ind.injury),
            })

    def to_dataframe(self) -> pd.DataFrame:
        """Собираем pandas-таблицу из накопленных строк (или пустую, если данных нет)."""
        if not self.rows:
            return pd.DataFrame(columns=[
                "tick","sex","age","color","speed","aggression","strength","injury"
            ])
        return pd.DataFrame(self.rows)
