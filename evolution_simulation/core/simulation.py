# core/simulation.py
"""Модуль, запускающий симуляцию во времени (без Environment и без флага брачного сезона)."""

from typing import List
import numpy as np

from .params import SimulationParams
from .entities import Organism
from .reproduction import seasonal_reproduction


class Simulation:
    """
    Управляет течением времени в модели:
      - step(): один «тик» жизни — обновляет выживание, травмы, возраст; удаляет умерших,
      - season_step(): сезонное размножение (каждые params.season_length тиков),
      - run(): крутит заданное число тиков.
    Все «ручки» берутся из SimulationParams.
    """

    def __init__(self, population: List[Organism], params: SimulationParams):
        """
        :param population: стартовый список живых особей (Organism)
        :param params: глобальные параметры симуляции (SimulationParams)
        """
        self.population: List[Organism] = population
        self.params: SimulationParams = params
        self.random_generator = np.random.default_rng(self.params.seed)
        self.tick: int = 0                          # общий счётчик тиков времени
        self.ticks_in_current_season: int = 0       # сколько тиков прошло в текущем «сезоне»

    def step(self) -> None:
        """
        Один «тик» времени:
          1) считаем вероятность выживания особи за тик,
          2) обновляем непрерывную травму (рост от агрессии и плотности столкновений; заживление),
          3) старим особь,
          4) удаляем умерших из памяти,
          5) если достигнута длина сезона — запускаем размножение.
        """
        self.tick += 1
        self.ticks_in_current_season += 1
        p = self.params

        # (0) Текущая плотность популяции: N/K (используем для частоты столкновений -> рост травм)
        alive_count_now: int = len(self.population)
        density_ratio: float = (alive_count_now / p.K) if p.K > 0 else 0.0  # N/K, если K>0

        next_population: List[Organism] = []

        for individual in self.population:
            # ---------------------------
            # (1) Вероятность выживания
            # ---------------------------
            # базовые слагаемые
            survival_probability = p.surv_base
            speed_cost_term = p.speed_surv_cost * (individual.speed ** 2)
            injury_cost_term = p.injury_survival_penalty * float(individual.injury)

            # возрастной риск
            age_excess = max(0, individual.age - p.age_mature)
            gompertz_term = p.age_gompertz_a * float(np.exp(p.age_gompertz_b * age_excess))

            # итог
            survival_probability *= float(np.exp(-(speed_cost_term + injury_cost_term + gompertz_term)))
            survival_probability = max(0.0, min(1.0, survival_probability))
            if individual.age >= p.age_hard_cap:
                survival_probability = 0.0


            # «Бросаем монетку» — выжил или нет
            if self.random_generator.random() > survival_probability:
                # особь умерла в этом тике — НЕ переносим её в новый список
                continue

            # ------------------------------------
            # (2) Обновляем непрерывную травму
            # ------------------------------------
            # Частота столкновений растёт с плотностью (N/K).
            # Чем выше агрессия, тем быстрее аккумулируется травма.
            encounter_rate = p.rho_enc_base * (0.5 + p.encounter_density_weight * max(0.0, density_ratio))

            # Рост травмы от агрессии и столкновений
            delta_injury = p.injury_severity_rate * max(0.0, individual.aggression) * encounter_rate
            # КАП, чтобы травма не «взрывалась» мгновенно
            delta_injury = min(delta_injury, p.injury_step_cap)

            updated_injury = float(individual.injury) + delta_injury
            # Рост травмы (агрессия * частота столкновений * скорость накопления)
            updated_injury += p.injury_severity_rate * max(0.0, individual.aggression) * encounter_rate
            updated_injury -= p.injury_recovery_rate
            individual.injury = max(0.0, min(1.0, updated_injury))

            # (3) Старение на 1 тик
            # ---------------------------
            individual.age += 1

            # Живую особь переносим в новый список
            next_population.append(individual)

        # --------------------------------------
        # (4) Компактификация: удаляем умерших
        # --------------------------------------
        self.population = next_population

        # ----------------------------------------------------
        # (5) Если сезон закончился — запускаем размножение
        # ----------------------------------------------------
        if self.ticks_in_current_season >= p.season_length:
            self.season_step()

    def season_step(self) -> None:
        """
        Сезонное размножение для всей популяции.
        Здесь же считаем коэффициент плотности и передаём его в reproduction.
        """
        p = self.params
        alive_count_now: int = len(self.population)
        density_ratio: float = (alive_count_now / p.K) if p.K > 0 else 0.0

        seasonal_reproduction(
            population=self.population,
            rng=self.random_generator,
            params=self.params,
            density_ratio=density_ratio,   # влияет на эффективную фертильность самок
        )

        # Начинаем новый «сезон» (сбрасываем локальный счётчик)
        self.ticks_in_current_season = 0

    def run(self, number_of_ticks: int) -> None:
        """Запускает симуляцию на заданное число тиков."""
        for _ in range(number_of_ticks):
            self.step()

    def population_size(self) -> int:
        """Возвращает текущее количество живых особей."""
        return len(self.population)
