# core/recorder.py
"""
TraitRecorder — простой сборщик срезов признаков во времени.
Идея: после каждого tика (или раз в несколько тиков) вызываем .snapshot(tick, species_list),
а потом строим графики по полученной таблице.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import pandas as pd
from .entities import Organism

@dataclass
class TraitRecorder:
    """
    Хранилище сырых записей (list of dict). Потом быстро превращаем в pandas.DataFrame.
    Здесь не анализируем данные — только копирование нужных полей.
    """
    # список строк (каждая — словарь колонок)
    rows: List[Dict[str, Any]] = field(default_factory=list)

    # core/recorder.py (фрагмент)
    def snapshot(self, tick: int, population: List[Organism]) -> None:
        for individ in population:
            if getattr(individ, "alive", True) is False:
                continue
            self.rows.append({
                "tick": tick,
                "sex": individ.sex,
                "age": individ.age,
                "injury": individ.injury,
                "color": individ.color,
                "speed": individ.speed,
                "lifestyle": individ.lifestyle,
                "activity": individ.activity,
                "aggression": individ.aggression,
                "strength": individ.strength,
            })


    def to_dataframe(self) -> pd.DataFrame:
        """Собираем pandas-таблицу из накопленных строк с признаками и их значениями."""

        if not self.rows: # если нет данных, высылаем предупреждение
            return "!!! Недостаточно данных для построения DataFrame !!!"
        
        return pd.DataFrame(self.rows)
