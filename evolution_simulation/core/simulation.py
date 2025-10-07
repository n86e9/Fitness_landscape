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
    def __init__(self, enviroment: Environment, species: List[Species]):
        self.enviroment = enviroment
        self.species = species
        self.random_generator = np.random.default_rng(enviroment.params.seed)
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
        self.enviroment.tick()
        parameters = self.enviroment.params

        for specie in self.species:
            for individual in specie.individuals:
                if not individual.alive: # если мёртв — пропускаем
                    continue

                surviveness = parameters.s_base - parameters.speed_survival_cost * (individual.speed ** 2) # рассчёт выживаемость
                surviveness = max(0.0, min(1.0, surviveness))

                if self.random_generator.random() > surviveness: 
                    individual.alive = False
                    continue

                if self.random_generator.random() < parameters.injury_rate:
                    individual.injury = True
                individual.age += 1

    def season_step(self) -> None:
        """
        Размножение по видам (одинаково простое для всех).
        Если захочешь запретить межвидовое скрещивание — оно и так отсутствует,
        потому что мы размножаем по каждому виду отдельно.
        """
        for specie in self.species:
            seasonal_reproduction_for_species(specie, self.random_generator, self.enviroment.params)
        # сброс травм к началу нового сезона (по желанию)
        
        for specie in self.species:
            for individ in specie.individuals:
                individ.injury = False

    def run(self, n_ticks: int) -> None:
        for _ in range(n_ticks):
            self.step()
            if self.tick % self.enviroment.params.season_length == 0:
                self.season_step()

    def population_size(self) -> int:
        return sum(specie.alive_count() for specie in self.species)
