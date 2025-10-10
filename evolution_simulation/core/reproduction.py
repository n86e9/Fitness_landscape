# core/reproduction.py
"""
Логика сезонного размножения (панмиксия, без Environment).
Здесь НЕТ формулы выживания — выживание считается в Simulation.step().
Тут мы:
  1) считаем эффективную ожидаемую фертильность самки на сезон λ_eff,
  2) выбираем отцов для каждого ребёнка по вероятностям (softmax от "привлекательности" самцов),
  3) рожаем детей: наследование (среднее родителей + сегрегационный шум) и редкие мутации.
"""

from typing import List
import numpy as np
from math import log1p

from .params import SimulationParams
from .entities import Organism


# ------------------------------ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ------------------------------

def _compute_effective_fertility_lambda_for_female(
    female: Organism,
    params: SimulationParams,
    density_ratio: float,
) -> float:
    """
    Эффективная ожидаемая фертильность самки за сезон:
      λ_eff = λ_base
              * F_density(N/K)
              * F_injury(injury_female)
              * F_age(age_female)

    где:
      - F_density = 1 / (1 + dens_alpha * (N/K))  — простая плотностная регуляция,
      - F_injury  = max(0, 1 - injury_fertility_penalty * injury_female) — травма снижает фертильность,
      - F_age     = exp(- ((age - age_peak)^2) / (2 * age_sigma^2)) — «колокол» (если age_sigma > 0).
    """
    # Плотностная регуляция — меньше детей при высокой плотности
    density_multiplier: float = 1.0 / (1.0 + params.dens_alpha * max(0.0, density_ratio))

    # Штраф за травму самки — линейный множитель
    fertility_injury_multiplier: float = max(
        0.0, 1.0 - params.injury_fertility_penalty * float(female.injury)
    )

    # Возрастной эффект фертильности (мягкий колокол). Можно «выключить», дав большую sigma.
    if getattr(params, "age_sigma", 0.0) and params.age_sigma > 0.0:
        age_deviation = (female.age - getattr(params, "age_peak", 3.0))
        age_effect = float(np.exp(- (age_deviation ** 2) / (2.0 * (params.age_sigma ** 2))))
    else:
        age_effect = 1.0

    lambda_effective: float = (
        params.lambda_base
        * density_multiplier
        * fertility_injury_multiplier
        * age_effect
    )
    return max(0.0, lambda_effective)


def _compute_male_attractiveness_scores(
    males: List[Organism],
    params: SimulationParams,
) -> np.ndarray:
    """
    Оценка «привлекательности» самцов для вероятностного выбора отца (softmax).

    Скорая (score_m) линейно-комбинирует сигналы качества и штрафы:
      score_m =
          + beta_color    * color
          + beta_strength * ln(1 + strength)
          + beta_speed    * ln(1 + speed)
          - beta_injury   * injury
          - beta_overaggr * max(0, aggression - aggr_star)

    Комментарии:
      - color здесь — «орнамент/яркость» (ты сам так решил), действует только у самцов.
      - ln(1+·) сглаживает влияние очень больших значений (убирает дикость хвостов).
      - injury снижает шансы быть выбранным отцом.
      - слишком высокая агрессия (выше порога aggr_star) штрафуется.
    """
    # Гарантируем наличие весов — иначе ставим нули/дефолты, чтобы не падало при неполном Params
    beta_color      = getattr(params, "beta_color",      0.0)
    beta_strength   = getattr(params, "beta_strength",   0.0)
    beta_speed      = getattr(params, "beta_speed",      0.0)
    beta_injury     = getattr(params, "beta_injury",     0.0)
    beta_overaggr   = getattr(params, "beta_overaggr",   0.0)
    aggr_star       = getattr(params, "aggr_star",       1e9)  # огромный порог => штрафа не будет, если не задан

    scores: List[float] = []
    for male in males:
        # Сигналы качества (все неотрицательные благодаря log1p)
        quality_term = (
            beta_color    * float(male.color) +
            beta_strength * log1p(max(0.0, male.strength)) +
            beta_speed    * log1p(max(0.0, male.speed))
        )
        # Штрафы
        injury_penalty = beta_injury * float(male.injury)
        overaggr_penalty = beta_overaggr * max(0.0, male.aggression - aggr_star)

        score_m = quality_term - injury_penalty - overaggr_penalty
        scores.append(score_m)

    return np.asarray(scores, dtype=float)


def make_child(
    mother: Organism,
    father: Organism,
    rng: np.random.Generator,
    params: SimulationParams
) -> Organism:
    """
    Рождение одного ребёнка:
      - наследование: признак ребёнка = среднее (мать, отец) + сегрегационный шум,
      - мутация: с вероятностью params.mutation добавляем гауссов сдвиг (σ = params.mutation_sigma),
      - пол 50/50; в конце жёстко приводим признаки к допустимым диапазонам.
    """

    def average_with_segregation_noise(a: float, b: float) -> float:
        """Среднее родителей + сегрегационный шум (Normal(0, seg_sigma))."""
        return 0.5 * (a + b) + rng.normal(0.0, params.seg_sigma)

    # Наследование по каждому признаку (без перекоса по полу)
    inherited_color      = average_with_segregation_noise(mother.color,      father.color)
    inherited_speed      = average_with_segregation_noise(mother.speed,      father.speed)
    inherited_lifestyle  = average_with_segregation_noise(mother.lifestyle,  father.lifestyle)
    inherited_activity   = average_with_segregation_noise(mother.activity,   father.activity)
    inherited_aggression = average_with_segregation_noise(mother.aggression, father.aggression)
    inherited_strength   = average_with_segregation_noise(mother.strength,   father.strength)

    def maybe_apply_mutation(value: float, clamp_01: bool = False, clamp_nonneg: bool = False) -> float:
        """
        С вероятностью params.mutation добавляем к значению гауссов сдвиг (σ = params.mutation_sigma).
        После этого, по необходимости, обрезаем:
          - clamp_01: в [0, 1]
          - clamp_nonneg: в [0, +∞)
        """
        if rng.random() < params.mutation:
            value += rng.normal(0.0, params.mutation_sigma)
        if clamp_01:
            value = float(min(1.0, max(0.0, value)))
        if clamp_nonneg:
            value = float(max(0.0, value))
        return value

    # Применяем мутации и обрезки по диапазонам
    child_color      = maybe_apply_mutation(inherited_color,      clamp_01=True)
    child_lifestyle  = maybe_apply_mutation(inherited_lifestyle,  clamp_01=True)
    child_activity   = maybe_apply_mutation(inherited_activity,   clamp_01=True)
    child_speed      = maybe_apply_mutation(inherited_speed,      clamp_nonneg=True)
    child_aggression = maybe_apply_mutation(inherited_aggression, clamp_nonneg=True)
    child_strength   = maybe_apply_mutation(inherited_strength,   clamp_nonneg=True)

    # Пол ребёнка — монетка 50/50
    child_sex = 'F' if rng.random() < 0.5 else 'M'

    # Создаём объект Organism (возраст=0, травма=0, живой=да)
    newborn = Organism(
        color=child_color,
        speed=child_speed,
        lifestyle=child_lifestyle,
        activity=child_activity,
        aggression=child_aggression,
        strength=child_strength,
        sex=child_sex,
        age=0,
        injury=0.0,
        alive=True
    )
    newborn.clamp()
    return newborn


# ------------------------------ ОСНОВНАЯ ФУНКЦИЯ СЕЗОНА ------------------------------

def seasonal_reproduction(
    population: List[Organism],
    rng: np.random.Generator,
    params: SimulationParams,
    density_ratio: float,
) -> None:
    """
    Сезонное размножение для ОДНОЙ популяции (панмиксия):

    1) Выделяем живых самок и самцов (на старте сезона Simulation уже удалил умерших).
    2) Считаем вероятность выбора каждого самца в качестве отца: softmax от score_m.
    3) Для каждой самки:
         - вычисляем её λ_eff (с учётом плотности, травмы и возраста),
         - сэмплируем число детей K ~ Poisson(λ_eff) (и ограничиваем максимумом),
         - K раз выбираем отца по вероятностям и рожаем ребёнка (наследование + мутации).
    4) Добавляем всех новорождённых в популяцию.
    """
    # Фактически все особи в population — живые, но оставим фильтр на случай, если ты где-то добавишь трупы
    female_list: List[Organism] = [ind for ind in population if getattr(ind, "alive", True) and ind.sex == 'F']
    male_list:   List[Organism] = [ind for ind in population if getattr(ind, "alive", True) and ind.sex == 'M']

    if not female_list or not male_list:
        return  # если нет самок или нет самцов — потомство не появляется

    # ---- 2) Вероятностный выбор отца: softmax по score --------------------------------
    male_scores = _compute_male_attractiveness_scores(male_list, params)
    # стабилизация чисел перед экспонентой
    male_scores -= male_scores.max()
    male_weights = np.exp(male_scores)
    total_weight = float(male_weights.sum())
    if total_weight <= 0.0:
        # если все оценки крайне плохие (веса ~0), то выбираем равномерно
        male_probabilities = np.ones_like(male_weights) / len(male_weights)
    else:
        male_probabilities = male_weights / total_weight

    # ---- 3) Генерация потомства --------------------------------------------------------
    newborns: List[Organism] = []

    for mother in female_list:
        # Эффективная фертильность самки за сезон (учитывает плотность/травму/возраст)
        lambda_effective = _compute_effective_fertility_lambda_for_female(
            female=mother,
            params=params,
            density_ratio=density_ratio,
        )

        # Сэмплируем количество детей
        number_of_children = int(rng.poisson(lam=lambda_effective))

        # «Крышуем» максимальное число детей на самку (страховка от взрывов)
        if params.max_children_per_female is not None:
            number_of_children = min(number_of_children, params.max_children_per_female)

        if number_of_children <= 0:
            continue  # эта самка в этом сезоне осталась без потомства

        # Для каждого ребёнка: выбираем отца по распределению и создаём ребёнка
        for _ in range(number_of_children):
            father = rng.choice(male_list, p=male_probabilities)
            child = make_child(mother, father, rng, params)
            newborns.append(child)

    # ---- 4) Пополнение популяции новорождёнными ---------------------------------------
    population.extend(newborns)
