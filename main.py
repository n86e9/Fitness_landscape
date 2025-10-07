# main.py
from evolution_simulation.core.parameters import SimulationParameters
from evolution_simulation.core.entities import Species, Organism
from evolution_simulation.core.environment import Environment
from evolution_simulation.core.simulation import Simulation
from evolution_simulation.core.recorder import TraitRecorder
from evolution_simulation.visualization.traits import plot_speed_hist_at_time

def make_species_A() -> Species:
    # маленькая стартовая популяция "вокруг" средних значений (без мудрёной математики)
    specie = Species(name="Species A")
    import numpy as np
    rng = np.random.default_rng(7)
    for _ in range(80):
        sex = 'F' if rng.random() < 0.5 else 'M'
        individ = Organism(
            color=float(np.clip(rng.normal(0.55, 0.06), 0, 1)),
            speed=float(max(0.0, rng.normal(2.0, 0.5))),
            lifestyle=float(np.clip(rng.normal(0.3, 0.1), 0, 1)),
            activity=float(np.clip(rng.normal(0.9, 0.08), 0, 1)),
            aggression=float(max(0.0, rng.normal(0.35, 0.15))),
            strength=float(max(0.0, rng.normal(1.1, 0.3))),
            sex=sex
        )
        individ.clamp()
        specie.individuals.append(individ)
    return specie

def make_species_B() -> Species:
    specie = Species(name="Species B")
    import numpy as np
    rng = np.random.default_rng(11)
    for _ in range(80):
        sex = 'F' if rng.random() < 0.5 else 'M'
        individ = Organism(
            color=float(np.clip(rng.normal(0.35, 0.06), 0, 1)),
            speed=float(max(0.0, rng.normal(1.3, 0.4))),
            lifestyle=float(np.clip(rng.normal(0.7, 0.1), 0, 1)),
            activity=float(np.clip(rng.normal(0.4, 0.1), 0, 1)),
            aggression=float(max(0.0, rng.normal(0.25, 0.12))),
            strength=float(max(0.0, rng.normal(0.9, 0.25))),
            sex=sex
        )
        individ.clamp()
        specie.individuals.append(individ)
    return specie

if __name__ == "__main__":
    # 1) параметры и среда
    params = SimulationParameters(season_length=10, s_base=0.985, speed_survival_cost=0.008,
                       lambda_base=1.5, max_children_per_female=3, seed=123)
    enviroment = Environment(background=0.55, risk_peak=1.0, params=params)

    # 2) виды и симуляция
    spA, spB = make_species_A(), make_species_B()
    simulation = Simulation(enviroment, [spA, spB])

    # 3) рекордер для графиков
    recorder = TraitRecorder()

    # 4) прогон: снимаем срез после каждого тика и после каждого сезона
    total_ticks = 50
    for _ in range(total_ticks):
        simulation.step()
        recorder.snapshot(simulation.tick, simulation.species)             # снимок после тика
        if simulation.tick % params.season_length == 0:
            simulation.season_step()
            recorder.snapshot(simulation.tick, simulation.species)         # снимок сразу после размножения

    print(f"t={simulation.tick}, N_total={simulation.population_size()}, "
          f"A={spA.alive_count()}, B={spB.alive_count()}")

    # 5) строим один простой график: гистограмма 'speed' на выбранном t
    df = recorder.to_dataframe()
    # выбери любой t, который точно есть в df (например, кратный season_length)
    plot_speed_hist_at_time(df, tick=10, bins=25, title="Speed hist at t=10")