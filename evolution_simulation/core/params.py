# evolution_simulation/core/params.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class SimulationParams:
    """
    Согласованный набор параметров:
    - выживание: surv_base, speed_surv_cost, injury_survival_penalty
    - травма: severity/recovery + rho_enc_base + cap
    - размножение: lambda_base + dens_alpha + max_children_per_female
    - наследование/мутации
    - стабилизирующий отбор (гора) по speed/aggression
    """

    # --- время / ёмкость ---
    season_length: int = 6 # тиков в сезоне
    K: float = 800.0 # ёмкость среды (макс. численность организмов)
    seed: Optional[int] = 123 # сид для генератора случайных чисел

    # --- выживание за тик ---
    surv_base: float = 0.994 # базовая вероятность выживания за тик
    speed_surv_cost: float = 0.003 # стоимость выживания за скорость
    injury_survival_penalty: float = 0.25 # штраф за травму
    age_hard_cap: int = 120 # жесткий предел возраста

    # --- травма ---
    injury_severity_rate: float = 0.010 # базовая скорость получения травмы
    injury_recovery_rate: float = 0.020 # базовая скорость восстановления травмы
    rho_enc_base: float = 1.0 / 90.0 # базовая скорость получения травмы
    injury_step_cap: float = 0.02 # ограничение на шаг травмы

    # --- размножение ---
    lambda_base: float = 1.30 # базовая фертильность
    dens_alpha: float = 3.0 # коэффициент плотностного ограничения при размножении
    max_children_per_female: int = 3 # макс. число детей за сезон на одну самку

    # ВАЖНО: это поле используется в reproduction.py и передаётся из Streamlit
    injury_fertility_penalty: float = 0.40

    # --- наследование и мутации ---
    seg_sigma: float = 0.05 # стандартное отклонение для наследования признаков
    mutation: float = 0.05 # вероятность мутации
    mutation_sigma: float = 0.04 # стандартное отклонение для мутаций

    # --- стабилизирующий отбор (гора) - делаем фильтрацию по признакам в процессе размножения ---
    opt_speed: float = 1.8 # оптимальное значение скорости
    opt_aggr: float  = 0.3 # оптимальное значение агрессии
    sel_sigma_speed: float = 0.6 # стандартное отклонение для селекции по скорости
    sel_sigma_aggr:  float = 0.4 # стандартное отклонение для селекции по агрессии
    sel_weight: float = 11.2 # вес селекции
