# core/reproduction.py
from typing import List
import numpy as np
from .parameters import SimulationParameters
from .entities import Organism, Species

def make_child(mother: Organism, father: Organism,
               rng: np.random.Generator, p: SimulationParameters) -> Organism:
    """
    Рождение одного ребёнка:
      признак = среднее родителей + маленький 'сегрегационный' шум;
      редкая мутация: ± небольшой гауссов сдвиг;
      пол 50/50; обрезка в допустимые диапазоны.
    """

    def avg(a, b): 
        return 0.5*(a+b) + rng.normal(0.0, p.seg_sigma)
    
    # базовое наследование
    color     = avg(mother.color, father.color)
    speed     = avg(mother.speed, father.speed)
    lifestyle = avg(mother.lifestyle, father.lifestyle)
    activity  = avg(mother.activity, father.activity)
    aggression= avg(mother.aggression, father.aggression)
    strength  = avg(mother.strength, father.strength)

    # редкие мутации
    def mutation(x, bounded01=False, nonneg=False):
        if rng.random() < p.mutaion:
            x += rng.normal(0.0, p.mutation_sigma)
        if bounded01: x = float(min(1.0, max(0.0, x)))
        if nonneg:    x = float(max(0.0, x))
        return x

    color      = mutation(color, bounded01=True)
    lifestyle  = mutation(lifestyle, bounded01=True)
    activity   = mutation(activity, bounded01=True)
    speed      = mutation(speed, nonneg=True)
    aggression = mutation(aggression, nonneg=True)
    strength   = mutation(strength, nonneg=True)

    sex = 'F' if rng.random() < 0.5 else 'M'
    baby = Organism(color, speed, lifestyle, activity, aggression, strength, sex)
    baby.clamp()
    return baby

def seasonal_reproduction_for_species(specie: Species,
                                      rng: np.random.Generator,
                                      p: SimulationParameters) -> None:
    """
    Простейшая сезонная репродукция для одного вида:
      - берём живых самок/самцов,
      - у каждой самки детей ~ Poisson(lambda_base) (ограничиваем),
      - отец выбирается равновероятно из живых самцов,
      - добавляем детей в тот же вид.
    """
    females = [x for x in specie.individuals if x.alive and x.sex == 'F']
    males   = [x for x in specie.individuals if x.alive and x.sex == 'M']
    if not females or not males:
        return

    newborns: List[Organism] = []
    for mom in females:
        k = int(rng.poisson(lam=p.lambda_base))
        if p.max_children_per_female is not None:
            k = min(k, p.max_children_per_female)
        for _ in range(k):
            dad = rng.choice(males)
            newborns.append(make_child(mom, dad, rng, p))

    specie.individuals.extend(newborns)
