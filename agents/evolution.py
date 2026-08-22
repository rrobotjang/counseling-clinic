"""
심리상담 에이전트 시스템 - 진화 알고리즘 고도화

达尔文 진화론 적용:
- 변이 (Mutation): 프롬프트/속성 변형
- 유전 (Inheritance): 부모→자손 특성 전달
- 자연선택 (Natural Selection): 환경 적응도 기반 선택
- 세대 교체 (Generational Turnover): 세대별 에이전트 교체
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import random
import json
import math


class AgentDNA:
    def __init__(self, prompt: str, traits: Dict[str, float] = None, generation: int = 1):
        self.prompt = prompt
        self.traits = traits or {
            "empathy": 0.5,
            "logic": 0.5,
            "energy": 1.0,
            "adaptability": 0.5,
            "social": 0.5,
            "aggression": 0.3,
            "cooperation": 0.5,
            "curiosity": 0.5
        }
        self.generation = generation
        self.created_at = datetime.now().isoformat()
        self.lineage: List[str] = []

    def mutate(self, rate: float = 0.1, intensity: float = 0.15) -> 'AgentDNA':
        new_traits = {}
        for trait, value in self.traits.items():
            if random.random() < rate:
                delta = random.gauss(0, intensity)
                new_traits[trait] = max(0.0, min(1.0, value + delta))
            else:
                new_traits[trait] = value

        prompt_variants = [
            self.prompt,
            f"당신은 {self._trait_summary(new_traits)} 성격을 가진 상담사입니다.",
            f"당신의 특성: {', '.join(f'{k}={v:.2f}' for k, v in new_traits.items())}"
        ]
        new_prompt = random.choice(prompt_variants)

        child = AgentDNA(new_prompt, new_traits, self.generation + 1)
        child.lineage = self.lineage.copy()
        return child

    def crossover(self, other: 'AgentDNA') -> 'AgentDNA':
        new_traits = {}
        weights = []

        for trait in self.traits:
            w = random.random()
            weights.append(w)
            if w < 0.5:
                new_traits[trait] = self.traits[trait]
            else:
                new_traits[trait] = other.traits.get(trait, self.traits[trait])

        blended_traits = {}
        for trait in new_traits:
            blend = random.uniform(0.3, 0.7)
            p1 = self.traits.get(trait, 0.5)
            p2 = other.traits.get(trait, 0.5)
            blended_traits[trait] = blend * p1 + (1 - blend) * p2
            blended_traits[trait] = max(0.0, min(1.0, blended_traits[trait]))

        new_gen = max(self.generation, other.generation) + 1
        prompt = random.choice([self.prompt, other.prompt])
        child = AgentDNA(prompt, blended_traits, new_gen)
        child.lineage = self.lineage.copy() + other.lineage.copy()
        return child

    def fitness(self, environment: Dict[str, float] = None) -> float:
        if environment is None:
            environment = {"social_pressure": 0.5, "danger_level": 0.3, "resource_scarcity": 0.4}

        base = (
            self.traits["empathy"] * 0.25 +
            self.traits["logic"] * 0.20 +
            self.traits["adaptability"] * 0.20 +
            self.traits["cooperation"] * 0.15 +
            self.traits["curiosity"] * 0.10 +
            self.traits["social"] * 0.10
        )

        env_bonus = 0.0
        if environment.get("social_pressure", 0) > 0.5:
            env_bonus += self.traits["social"] * 0.1
        if environment.get("danger_level", 0) > 0.5:
            env_bonus += self.traits["logic"] * 0.1
        if environment.get("resource_scarcity", 0) > 0.5:
            env_bonus += self.traits["cooperation"] * 0.1

        return max(0.0, min(1.0, base + env_bonus))

    def _trait_summary(self, traits: Dict[str, float]) -> str:
        top = sorted(traits.items(), key=lambda x: x[1], reverse=True)[:3]
        trait_names = {
            "empathy": "공감적인", "logic": "논리적인", "adaptability": "적응력이 뛰어난",
            "social": "사교적인", "cooperation": "협력적인", "curiosity": "호기심 많은",
            "aggression": "공격적인"
        }
        return "과 ".join(trait_names.get(k, k) for k, _ in top)

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "traits": self.traits,
            "generation": self.generation,
            "lineage_length": len(self.lineage),
            "created_at": self.created_at
        }


class VirtualAgent:
    def __init__(self, agent_id: str, name: str, dna: AgentDNA):
        self.agent_id = agent_id
        self.name = name
        self.dna = dna
        self.position = {"x": random.randint(50, 750), "y": random.randint(50, 550)}
        self.energy = 100.0
        self.is_alive = True
        self.age = 0
        self.interaction_count = 0
        self.created_at = datetime.now()
        self.last_active = datetime.now()

    def respond(self, message: str) -> str:
        empathy = self.dna.traits["empathy"]
        logic = self.dna.traits["logic"]
        social = self.dna.traits["social"]

        if empathy > 0.7:
            prefix = "공감적으로 말하면,"
        elif logic > 0.7:
            prefix = "논리적으로 생각하면,"
        elif social > 0.7:
            prefix = "함께 생각해보면,"
        else:
            prefix = "생각해보니,"

        if self.energy > 70:
            quality = "좋은 생각이 있습니다"
        elif self.energy > 40:
            quality = "조금 생각이 필요합니다"
        else:
            quality = "에너지가 부족하지만 노력하겠습니다"

        response = f"{prefix} \"{message[:50]}\"에 대해 {quality}."
        self.interaction_count += 1
        self.update_energy(-2)
        self.last_active = datetime.now()
        return response

    def update_energy(self, delta: float):
        self.energy = max(0.0, min(100.0, self.energy + delta))

    def age_step(self):
        self.age += 1
        decay = self.dna.traits["adaptability"] * 2
        self.update_energy(-decay)

    def check_survival(self) -> bool:
        if self.energy <= 0 or self.age > 200:
            self.is_alive = False
        return self.is_alive

    def get_state(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "position": self.position,
            "energy": round(self.energy, 1),
            "age": self.age,
            "is_alive": self.is_alive,
            "generation": self.dna.generation,
            "traits": {k: round(v, 3) for k, v in self.dna.traits.items()},
            "fitness": round(self.dna.fitness(), 3),
            "interactions": self.interaction_count
        }


class EvolutionEngine:
    def __init__(self):
        self.mutation_rate = 0.15
        self.mutation_intensity = 0.12
        self.selection_pressure = 0.4
        self.crossover_rate = 0.6
        self.environment = {
            "social_pressure": 0.5,
            "danger_level": 0.3,
            "resource_scarcity": 0.4
        }
        self.generation = 1
        self.history: List[dict] = []

    def evolve(self, agents: List[VirtualAgent]) -> Tuple[List[VirtualAgent], dict]:
        alive = [a for a in agents if a.is_alive]
        if len(alive) < 2:
            return agents, {"status": "insufficient_agents", "count": len(alive)}

        stats_before = self._population_stats(alive)

        survivors = self._selection(alive)
        offspring = self._reproduction(survivors)
        self._mutate_population(offspring)
        next_gen = self._next_generation(alive, survivors, offspring)

        stats_after = self._population_stats(next_gen)
        self.generation += 1

        evolution_record = {
            "generation": self.generation,
            "before": stats_before,
            "after": stats_after,
            "survivors": len(survivors),
            "offspring": len(offspring),
            "timestamp": datetime.now().isoformat()
        }
        self.history.append(evolution_record)

        return next_gen, evolution_record

    def _selection(self, agents: List[VirtualAgent]) -> List[VirtualAgent]:
        scored = [(a, a.dna.fitness(self.environment)) for a in agents]
        scored.sort(key=lambda x: x[1], reverse=True)

        n_survive = max(2, int(len(scored) * (1 - self.selection_pressure)))
        return [a for a, _ in scored[:n_survive]]

    def _reproduction(self, survivors: List[VirtualAgent]) -> List[VirtualAgent]:
        offspring = []

        for i in range(len(survivors)):
            if random.random() < self.crossover_rate and len(survivors) > 1:
                partner_idx = random.choice([j for j in range(len(survivors)) if j != i])
                child_dna = survivors[i].dna.crossover(survivors[partner_idx].dna)
            else:
                child_dna = AgentDNA(
                    survivors[i].dna.prompt,
                    survivors[i].dna.traits.copy(),
                    survivors[i].dna.generation
                )
                child_dna.lineage = survivors[i].dna.lineage.copy()

            child_id = f"agent-g{self.generation}-{random.randint(1000, 9999)}"
            child_name = f"G{child_dna.generation}-{child_id[-4:]}"
            child = VirtualAgent(child_id, child_name, child_dna)
            child.energy = 80.0
            offspring.append(child)

        return offspring

    def _mutate_population(self, agents: List[VirtualAgent]):
        for agent in agents:
            agent.dna = agent.dna.mutate(self.mutation_rate, self.mutation_intensity)

    def _next_generation(self, current: List[VirtualAgent], survivors: List[VirtualAgent], offspring: List[VirtualAgent]) -> List[VirtualAgent]:
        for agent in current:
            if agent not in survivors:
                agent.check_survival()

        alive_survivors = [a for a in survivors if a.is_alive]
        return alive_survivors + offspring

    def _population_stats(self, agents: List[VirtualAgent]) -> dict:
        if not agents:
            return {"count": 0, "avg_fitness": 0, "avg_energy": 0, "avg_generation": 0, "traits_avg": {}}

        fitnesses = [a.dna.fitness(self.environment) for a in agents]
        energies = [a.energy for a in agents]
        generations = [a.dna.generation for a in agents]

        traits_avg = {}
        for trait in agents[0].dna.traits:
            values = [a.dna.traits.get(trait, 0) for a in agents]
            traits_avg[trait] = round(sum(values) / len(values), 3)

        diversity = self._calc_diversity(agents)

        return {
            "count": len(agents),
            "avg_fitness": round(sum(fitnesses) / len(fitnesses), 3),
            "max_fitness": round(max(fitnesses), 3),
            "min_fitness": round(min(fitnesses), 3),
            "avg_energy": round(sum(energies) / len(energies), 1),
            "avg_generation": round(sum(generations) / len(generations), 1),
            "traits_avg": traits_avg,
            "diversity": round(diversity, 3)
        }

    def _calc_diversity(self, agents: List[VirtualAgent]) -> float:
        if len(agents) < 2:
            return 0.0

        total_variance = 0.0
        trait_count = 0

        for trait in agents[0].dna.traits:
            values = [a.dna.traits.get(trait, 0) for a in agents]
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            total_variance += variance
            trait_count += 1

        return math.sqrt(total_variance / trait_count) if trait_count > 0 else 0.0

    def set_environment(self, env: Dict[str, float]):
        self.environment.update(env)

    def get_history(self) -> List[dict]:
        return self.history

    def get_darwin_report(self) -> dict:
        if not self.history:
            return {"status": "no_data"}

        latest = self.history[-1]
        first = self.history[0]

        fitness_trend = latest["after"]["avg_fitness"] - first["after"]["avg_fitness"]
        diversity_trend = latest["after"].get("diversity", 0) - first["after"].get("diversity", 0)

        return {
            "total_generations": self.generation,
            "current_population": latest["after"]["count"],
            "fitness_change": round(fitness_trend, 3),
            "diversity_change": round(diversity_trend, 3),
            "traits_evolution": latest["after"]["traits_avg"],
            "conclusion": self._darwin_conclusion(fitness_trend, diversity_trend, latest)
        }

    def _darwin_conclusion(self, fitness_trend: float, diversity_trend: float, latest: dict) -> str:
        if fitness_trend > 0.05:
            fit = "적응도 상승 — 자연선택이 작동 중"
        elif fitness_trend < -0.05:
            fit = "적응도 하락 — 환경 불일치 또는 과도한 변이"
        else:
            fit = "적응도 안정 — 평형 상태"

        if diversity_trend > 0.02:
            div = "다양성 증가 — 변이가 다양성을 유지"
        elif diversity_trend < -0.02:
            div = "다양성 감소 — 선택 압력이 동종교배 유발"
        else:
            div = "다양성 안정"

        return f"{fit}. {div}."


class AgentManager:
    def __init__(self):
        self.agents: Dict[str, VirtualAgent] = {}
        self.evolution_engine = EvolutionEngine()
        self._create_default_agents()

    def _create_default_agents(self):
        configs = [
            ("counselor-1", "상담사 A", "공감적이고 따뜻한 상담사", {"empathy": 0.8, "logic": 0.4, "energy": 1.0, "adaptability": 0.6, "social": 0.7, "aggression": 0.1, "cooperation": 0.8, "curiosity": 0.5}),
            ("counselor-2", "상담사 B", "논리적이고 분석적인 상담사", {"empathy": 0.4, "logic": 0.8, "energy": 1.0, "adaptability": 0.6, "social": 0.5, "aggression": 0.2, "cooperation": 0.6, "curiosity": 0.6}),
            ("observer-1", "관찰자", "호기심 많은 관찰자", {"empathy": 0.5, "logic": 0.5, "energy": 1.0, "adaptability": 0.8, "social": 0.4, "aggression": 0.1, "cooperation": 0.5, "curiosity": 0.9}),
        ]
        for aid, name, prompt, traits in configs:
            dna = AgentDNA(prompt, traits)
            self.agents[aid] = VirtualAgent(aid, name, dna)

    def create_agent(self, agent_id: str, name: str, prompt: str, traits: Dict[str, float] = None) -> VirtualAgent:
        dna = AgentDNA(prompt, traits)
        agent = VirtualAgent(agent_id, name, dna)
        self.agents[agent_id] = agent
        return agent

    def get_agent(self, agent_id: str) -> Optional[VirtualAgent]:
        return self.agents.get(agent_id)

    def get_all_agents(self) -> List[dict]:
        return [a.get_state() for a in self.agents.values() if a.is_alive]

    def process_message(self, agent_id: str, message: str) -> str:
        agent = self.get_agent(agent_id)
        if agent and agent.is_alive:
            return agent.respond(message)
        return "에이전트를 찾을 수 없거나 비활성 상태입니다."

    def run_evolution(self) -> dict:
        alive = [a for a in self.agents.values() if a.is_alive]
        if len(alive) < 2:
            return {"status": "need_more_agents", "alive": len(alive)}

        new_agents, record = self.evolution_engine.evolve(alive)

        for agent in new_agents:
            if agent.agent_id not in self.agents:
                self.agents[agent.agent_id] = agent

        dead = [a for a in self.agents.values() if not a.is_alive]
        for d in dead:
            if d.age > 200:
                del self.agents[d.agent_id]

        return record

    def age_all_agents(self):
        for agent in list(self.agents.values()):
            if agent.is_alive:
                agent.age_step()

    def get_darwin_report(self) -> dict:
        return self.evolution_engine.get_darwin_report()

    def get_stats(self) -> dict:
        alive = [a for a in self.agents.values() if a.is_alive]
        dead_count = sum(1 for a in self.agents.values() if not a.is_alive)

        return {
            "total_ever_created": len(self.agents) + dead_count,
            "alive_agents": len(alive),
            "dead_agents": dead_count,
            "generation": self.evolution_engine.generation,
            "avg_energy": round(sum(a.energy for a in alive) / len(alive), 1) if alive else 0,
            "avg_fitness": round(sum(a.dna.fitness(self.evolution_engine.environment) for a in alive) / len(alive), 3) if alive else 0,
            "environment": self.evolution_engine.environment
        }
