# core/recorder.py
"""
TraitRecorder — простой сборщик срезов признаков во времени.
Идея: после каждого tика (или раз в несколько тиков) вызываем .snapshot(t, species_list),
а потом строим графики по полученной таблице.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import pandas as pd
from .entities import Species

@dataclass
class TraitRecorder:
    """
    Хранилище сырых записей (list of dict). Потом быстро превращаем в pandas.DataFrame.
    Здесь не делаем математику — только копирование нужных полей.
    """
    rows: List[Dict[str, Any]] = field(default_factory=list)

    def snapshot(self, t: int, species_list: List[Species]) -> None:
        """
        Снять срез на момент времени t.
        Для каждой живой особи сохраняем: вид, пол, возраст и ключевые признаки.
        Гибко: если нужно больше колонок — просто добавь ниже.
        """
        for sp in species_list:
            for ind in sp.individuals:
                if not ind.alive:
                    continue
                self.rows.append({
                    "t": t,
                    "species": sp.name,
                    "sex": ind.sex,
                    "age": ind.age,
                    "color": ind.color,
                    "speed": ind.speed,
                    "lifestyle": ind.lifestyle,
                    "activity": ind.activity,
                    "aggression": ind.aggression,
                    "strength": ind.strength,
                })

    def to_dataframe(self) -> pd.DataFrame:
        """Собрать pandas-таблицу из накопленных строк."""
        if not self.rows:
            return pd.DataFrame(columns=[
                "t","species","sex","age","color","speed","lifestyle","activity","aggression","strength"
            ])
        return pd.DataFrame(self.rows)
