from typing import List
import numpy as np
from .params import SimulationParams
from .entities import Organism

def selection_score(speed: float, aggression: float, p: SimulationParams) -> float:
    """
    Гауссов холм по (speed, aggression): максимум в (opt_speed, opt_aggr).
    Возвращает множитель ∈ (0, 1].
    """
    ds = (speed - p.opt_speed)
    da = (aggression - p.opt_aggr)
    ss = max(p.sel_sigma_speed, 1e-6)
    sa = max(p.sel_sigma_aggr,  1e-6)
    z = (ds*ds)/(2.0*ss*ss) + (da*da)/(2.0*sa*sa)
    return float(np.exp(- p.sel_weight * z))

def make_child(mother: Organism, father: Organism,
               rng: np.random.Generator, params: SimulationParams) -> Organism:
    """
    Рождение одного ребёнка:
      trait_child = 0.5*(mom + dad) + N(0, seg_sigma), редкие мутации, обрезка диапазонов.
    """
    def avg(a, b): 
        return 0.5 * (a + b) + rng.normal(0.0, params.seg_sigma)

    color      = avg(mother.color, father.color)
    speed      = avg(mother.speed, father.speed)
    lifestyle  = avg(mother.lifestyle, father.lifestyle)
    activity   = avg(mother.activity, father.activity)
    aggression = avg(mother.aggression, father.aggression)
    strength   = avg(mother.strength, father.strength)

    def mutate(x, limit_01: bool=False, nonneg: bool=False) -> float:
        if rng.random() < params.mutation:
            x += rng.normal(0.0, params.mutation_sigma)
        if limit_01:
            x = float(min(1.0, max(0.0, x)))
        if nonneg:
            x = float(max(0.0, x))
        return x

    color      = mutate(color, limit_01=True)
    lifestyle  = mutate(lifestyle, limit_01=True)
    activity   = mutate(activity, limit_01=True)
    speed      = mutate(speed, nonneg=True)
    aggression = mutate(aggression, nonneg=True)
    strength   = mutate(strength, nonneg=True)

    sex = 'F' if rng.random() < 0.5 else 'M'
    baby = Organism(color, speed, lifestyle, activity, aggression, strength, sex)
    baby.clamp()
    return baby

def seasonal_reproduction(population: List[Organism],
                          rng: np.random.Generator,
                          params: SimulationParams,
                          dens_ratio: float) -> None:
    """
    Раз в сезон:
      - плотностная регуляция λ,
      - фертильность матери умножаем на селекционный множитель (гора),
      - отцов выбираем по весам (та же гора),
      - ограничиваем детей per female.
    """
    females = [x for x in population if x.sex == 'F']
    males   = [x for x in population if x.sex == 'M']
    if not females or not males:
        return

    # мягкая плотностная регуляция
    f_dens = 1.0 / (1.0 + params.dens_alpha * max(0.0, dens_ratio))

    # веса мужчин для выбора отца (сексуальный отбор): ближе к оптимуму — чаще выбран
    male_weights = np.array([selection_score(m.speed, m.aggression, params) for m in males], dtype=float)
    if male_weights.sum() <= 0.0:
        male_weights = np.ones_like(male_weights)
    male_probs = male_weights / male_weights.sum()

    newborns: List[Organism] = []

    for mom in females:
        mom_sel = selection_score(mom.speed, mom.aggression, params)  # селекция матери
        mom_penalty = (1.0 - params.injury_fertility_penalty * float(mom.injury))  # травма снижает фертильность
        lam_eff = max(0.0, params.lambda_base * f_dens * mom_sel * mom_penalty)

        k = int(rng.poisson(lam=lam_eff))
        if params.max_children_per_female is not None:
            k = min(k, params.max_children_per_female)
        if k <= 0:
            continue

        # выбираем отцов по вероятностям (replace=True — один отец может участвовать многократно)
        dad_idx = rng.choice(len(males), size=k, replace=True, p=male_probs)
        for j in dad_idx:
            dad = males[int(j)]
            newborns.append(make_child(mom, dad, rng, params))

    population.extend(newborns)
