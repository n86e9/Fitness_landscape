# evolution_simulation/core/simulation.py
from typing import List
import numpy as np
from .params import SimulationParams
from .entities import Organism
from .reproduction import seasonal_reproduction

def selection_score(speed: float, aggression: float, p: SimulationParams) -> float:
    """
    Гауссов холм по (speed, aggression): максимум в (opt_speed, opt_aggr).
    Возвращает множитель ∈ (0, 1], чем ближе к оптимуму — тем ближе к 1.
    """
    ds = (speed - p.opt_speed)
    da = (aggression - p.opt_aggr)
    ss = max(p.sel_sigma_speed, 1e-6)  # защита от 0
    sa = max(p.sel_sigma_aggr,  1e-6)
    z = (ds*ds)/(2.0*ss*ss) + (da*da)/(2.0*sa*sa)
    return float(np.exp(- p.sel_weight * z))

class Simulation:
    """
    Управляет временем. Простейшая смертность + травма + старение.
    Размножение — раз в season_length тиков.
    """
    def __init__(self, population: List[Organism], params: SimulationParams):
        self.population = population
        self.params = params
        self.tick = 0
        self.random_generator = np.random.default_rng(params.seed)

    def step(self) -> None:
        p = self.params
        self.tick += 1

        next_pop: List[Organism] = []

        for ind in self.population:
            # ---------- 1) ВЫЖИВАЕМОСТЬ ЗА ТИК ----------
            # ЮВЕНИЛЬНЫЙ БУФЕР: до 3 тиков штрафы мягче
            is_juvenile = (ind.age < 3)

            # коэффициенты штрафов С УЧЁТОМ ювенильного буфера
            speed_cost  = (0.5 * p.speed_surv_cost if is_juvenile else p.speed_surv_cost)
            inj_penalty = (0.5 * p.injury_survival_penalty if is_juvenile else p.injury_survival_penalty)

            # слагаемые экспоненты
            speed_term  = speed_cost * (ind.speed ** 2)
            injury_term = inj_penalty * float(ind.injury)

            # базовая выживаемость (без отбора)
            base_surv = p.surv_base * float(np.exp(-(speed_term + injury_term)))

            # стабилизирующий отбор (пик вблизи opt_speed/opt_aggr)
            sel_mult  = selection_score(ind.speed, ind.aggression, p)

            # итоговая вероятность выжить за тик
            s_tick = float(np.clip(base_surv * sel_mult, 0.0, 1.0))

            # жёсткий потолок возраста
            if ind.age >= getattr(p, "age_hard_cap", 120):
                s_tick = 0.0

            # выжил?
            if self.random_generator.random() > s_tick:
                continue  # умер — не переносим

            # ---------- 2) ТРАВМА ----------
            # injury += severity_rate * aggression * rho_enc_base  (детёныши копят медленнее)
            # injury -= recovery_rate
            delta = p.injury_severity_rate * max(0.0, ind.aggression) * p.rho_enc_base
            if is_juvenile:
                delta *= 0.5
            if hasattr(p, "injury_step_cap"):
                delta = min(delta, p.injury_step_cap)

            new_injury = float(ind.injury) + delta - p.injury_recovery_rate
            ind.injury = float(np.clip(new_injury, 0.0, 1.0))

            # ---------- 3) СТАРЕНИЕ ----------
            ind.age += 1

            next_pop.append(ind)

        # обновляем популяцию
        self.population = next_pop

        # ---------- 4) РАЗМНОЖЕНИЕ ----------
        if self.tick % p.season_length == 0 and len(self.population) > 1:
            dens_ratio = (len(self.population) / p.K) if p.K > 0 else 0.0
            seasonal_reproduction(
                population=self.population,
                rng=self.random_generator,
                params=p,
                dens_ratio=dens_ratio
            )

    def population_size(self) -> int:
        return len(self.population)
