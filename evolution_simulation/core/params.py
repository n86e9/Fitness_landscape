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
    season_length: int = 6
    K: float = 800.0
    seed: Optional[int] = 123

    # --- выживание за тик ---
    surv_base: float = 0.994
    speed_surv_cost: float = 0.003
    injury_survival_penalty: float = 0.25
    age_hard_cap: int = 120

    # --- травма ---
    injury_severity_rate: float = 0.010
    injury_recovery_rate: float = 0.020
    rho_enc_base: float = 1.0 / 90.0
    injury_step_cap: float = 0.02

    # --- размножение ---
    lambda_base: float = 1.30
    dens_alpha: float = 3.0
    max_children_per_female: int = 3
    # ВАЖНО: это поле используется в reproduction.py и передаётся из Streamlit
    injury_fertility_penalty: float = 0.40

    # --- наследование и мутации ---
    seg_sigma: float = 0.05
    mutation: float = 0.05
    mutation_sigma: float = 0.04

    # --- стабилизирующий отбор (гора) ---
    opt_speed: float = 1.8
    opt_aggr: float  = 0.3
    sel_sigma_speed: float = 0.6
    sel_sigma_aggr:  float = 0.4
    sel_weight: float = 1.2
