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
    rows: List[Dict[str, Any]] = field(default_factory=list) # список строк (каждая — словарь колонок)

    def snapshot(self, tick: int, species_list: List[Species]) -> None:
        """
        Снять срез на момент времени t.
        Для каждой живой особи сохраняем: вид, пол, возраст и ключевые признаки.
        Гибко: если нужно больше колонок — просто добавь ниже.
        """
        for specie in species_list:
            for individ in specie.individuals:
                if not individ.alive:
                    continue
                self.rows.append({
                    "tick": tick,
                    "species": specie.name,
                    "sex": individ.sex,
                    "age": individ.age,
                    "color": individ.color,
                    "speed": individ.speed,
                    "lifestyle": individ.lifestyle,
                    "activity": individ.activity,
                    "aggression": individ.aggression,
                    "strength": individ.strength,
                })

    def to_dataframe(self) -> pd.DataFrame:
        """Собрать pandas-таблицу из накопленных строк."""
        if not self.rows:
            return pd.DataFrame(columns=[
                "tick","species","sex","age","color","speed","lifestyle","activity","aggression","strength"
            ])
        return pd.DataFrame(self.rows)
