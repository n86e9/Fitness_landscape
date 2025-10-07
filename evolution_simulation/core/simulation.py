# core/simulation.py
from typing import List
import numpy as np
from .parameters import SimulationParameters
from .entities import Species
from .environment import Environment
from .reproduction import seasonal_reproduction_for_species

class Simulation:
    """
    Управляет временем. Здесь простейшая смертность и возраст.
    Размножение — вызываем из reproduction.py на сезонных шагах.
    """
    def __init__(self, env: Environment, species: List[Species]):
        self.env = env
        self.species = species
        self.rng = np.random.default_rng(env.params.seed)
        self.tick = 0

    def step(self) -> None:
        """
        Один тик:
          - среда тикает,
          - простая смертность: s = s_base - c*speed^2,
          - случайная травма,
          - age += 1.
        Никакой «сложной математики».
        """
        self.tick += 1
        self.env.tick()
        p = self.env.params

        for sp in self.species:
            for ind in sp.individuals:
                if not ind.alive:
                    continue
                s = p.s_base - p.speed_survival_cost * (ind.speed ** 2)
                s = max(0.0, min(1.0, s))
                if self.rng.random() > s:
                    ind.alive = False
                    continue
                if self.rng.random() < p.injury_rate:
                    ind.injury = True
                ind.age += 1

    def season_step(self) -> None:
        """
        Размножение по видам (одинаково простое для всех).
        Если захочешь запретить межвидовое скрещивание — оно и так отсутствует,
        потому что мы размножаем по каждому виду отдельно.
        """
        for sp in self.species:
            seasonal_reproduction_for_species(sp, self.rng, self.env.params)
        # сброс травм к началу нового сезона (по желанию)
        for sp in self.species:
            for ind in sp.individuals:
                ind.injury = False

    def run(self, n_ticks: int) -> None:
        for _ in range(n_ticks):
            self.step()
            if self.tick % self.env.params.season_length == 0:
                self.season_step()

    def population_size(self) -> int:
        return sum(sp.alive_count() for sp in self.species)
