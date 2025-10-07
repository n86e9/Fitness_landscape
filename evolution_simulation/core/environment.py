# core/enviroment.py
from dataclasses import dataclass, field
from .parameters import SimulationParameters

@dataclass
class Environment:
    """
    Минимальная среда: фон (для окраса), пик риска (для активности),
    и ресурсы — пока без формул.
    """
    background: float = 0.5
    risk_peak: float = 1.0
    resource: float = 1000.0
    regen_rate: float = 50.0
    params: SimulationParameters = field(default_factory=SimulationParameters)

    def tick(self) -> None:
        # простое «восстановление» ресурса как пример динамики среды
        self.resource += self.regen_rate
