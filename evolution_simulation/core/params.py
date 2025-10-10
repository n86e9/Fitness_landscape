# core/params.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class SimulationParams:
    """
    Единый набор «ручек» модели. Поля названы максимально явно и согласованы
    с текущими модулями simulation.py, reproduction.py и main.py.
    """

    # --- Время и ёмкость среды ---
    season_length: int = 20              # длина «сезона» в тиках (после него размножение)
    K: float = 500.0                     # ёмкость среды (масштаб плотности N/K)
    seed: Optional[int] = 42             # сид RNG для воспроизводимости

    # --- Выживание за тик (s_tick) ---
    surv_base: float = 0.98              # базовый шанс выжить за тик до штрафов
    speed_surv_cost: float = 0.01        # штраф за скорость: s *= exp(-speed_surv_cost * speed^2)

    # --- столкновения и травма (демпферы) ---
    encounter_density_weight: float = 0.5  # сколько веса даём плотности в частоте встреч (0..1)
    injury_step_cap: float = 0.03          # максимум прироста травмы за тик (страховка)

    # --- старение (Гомпертц-Мейкхэм) ---
    age_mature: int = 2                 # с какого возраста считать старение (ювенильные тики без штрафа)
    age_gompertz_a: float = 0.002       # базовый уровень возрастного риска (меньше — дольше живут)
    age_gompertz_b: float = 0.12        # скорость экспоненциального роста риска с возрастом
    age_hard_cap: int = 120             # «жёсткий потолок» возраста (на этом тике смерть гарантирована)

    # --- Травма (непрерывная [0..1]) ---
    injury_severity_rate: float = 0.02   # скорость накопления травмы от агрессии/столкновений
    injury_recovery_rate: float = 0.01   # заживление травмы за тик
    injury_survival_penalty: float = 0.3 # множитель в выживании: s *= (1 - penalty * injury)
    injury_fertility_penalty: float = 0.5 # множитель в фертильности: λ *= (1 - penalty * injury)

    # Частота столкновений (масштабирует рост травмы)
    rho_enc_base: float = 1.0 / 50.0     # базовая «частота встреч» (чем больше, тем больше травм)

    # Половые коэффициенты влияния агрессии на травму
    male_aggression_injury_factor: float = 1.0   # у самцов травма растёт сильнее
    female_aggression_injury_factor: float = 0.3 # у самок слабее

    # Влияние «образа жизни» (lifestyle)
    lifestyle_conflict_damp: float = 0.5  # снижает столкновения: encounter *= (1 - damp * lifestyle)
    lifestyle_survival_bonus: float = 0.05# бонус к выживанию: s *= exp(+bonus * lifestyle)

    # --- Размножение (за сезон) ---
    lambda_base: float = 1.2              # среднее число детёнышей у самки до модификаторов
    max_children_per_female: int = 4      # кап на детей у самки (None — без ограничения)

    # Плотностная регуляция размножения
    dens_alpha: float = 2.0               # λ_eff *= 1/(1 + dens_alpha * (N/K))

    # Возрастная кривая фертильности самок (мягкий «колокол»; отключи, дав большую sigma=0)
    age_peak: float = 3.0                 # возраст максимальной фертильности
    age_sigma: float = 2.0                # ширина «колокола» (0 -> без эффекта)

    # --- Наследование и мутации у новорождённых ---
    seg_sigma: float = 0.05               # сегрегационный шум при усреднении родителей
    mutation: float = 0.05                # вероятность мутации на признак
    mutation_sigma: float = 0.04          # амплитуда гауссовой мутации

    # --- Сексуальный отбор (оценка привлекательности самцов) ---
    beta_color: float = 0.8               # вклад окраса (привлекательности)
    beta_strength: float = 1.0            # вклад силы (лог-сглаживание ln(1+strength))
    beta_speed: float = 0.6               # вклад скорости (ln(1+speed))
    beta_injury: float = 1.0              # штраф за травму
    beta_overaggr: float = 0.5            # штраф за «перегиб» агрессии
    aggr_star: float = 0.8                # порог для «перегиба» агрессии
    beta_lifestyle_in_males: float = 0.0  # (опц.) вклад lifestyle в привлекательность самцов
