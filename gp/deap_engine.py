"""DeapGeneticEngine — DEAP-based implementation of IGeneticEngine.

Provides the full GP pipeline: primitive registration, population
initialisation, evolutionary operators, and training loop.
"""

from __future__ import annotations

import functools
import operator
import random
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from deap import algorithms, base, creator, gp, tools

from gp.interface import GPResult, IGeneticEngine
from gp.primitives import (
    FUNCTION_SET,
    TERMINAL_NAMES,
    add,
    if_positive,
    mul,
    neg,
    protected_div,
    safe_max,
    safe_min,
    sub,
)


class DeapGeneticEngine(IGeneticEngine):
    """Concrete GP engine backed by DEAP.

    Lifecycle:
      1. ``setup(terminal_names, ...)`` — register primitives, create toolbox
      2. ``train(fitness_function, seed)`` — run the evolutionary algorithm
      3. ``evaluate_individual(ind, terminal_values)`` — score at runtime
    """

    def __init__(self) -> None:
        self._pset: Optional[gp.PrimitiveSetTyped] = None
        self._toolbox: Optional[base.Toolbox] = None
        self._terminal_names: List[str] = []
        self._params: Dict[str, Any] = {}
        self._setup_done = False
        self._compiled_cache: Dict[str, Any] = {}

    # ── IGeneticEngine interface ─────────────────────────────────────

    @property
    def name(self) -> str:
        return "deap"

    def setup(
        self,
        terminal_names: Optional[List[str]] = None,
        function_set: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Configure the DEAP primitive set, toolbox, and GP parameters.

        Keyword args (kwargs) map to GPConfig fields:
            population_size, n_generations, tournament_size,
            crossover_prob, mutation_prob, max_tree_depth,
            elitism_ratio, parsimony_coefficient, multi_objective
        """
        self._terminal_names = terminal_names or list(TERMINAL_NAMES)
        func_names = function_set or list(FUNCTION_SET.keys())
        self._params = {
            "population_size": kwargs.get("population_size", 150),
            "n_generations": kwargs.get("n_generations", 50),
            "tournament_size": kwargs.get("tournament_size", 3),
            "crossover_prob": kwargs.get("crossover_prob", 0.8),
            "mutation_prob": kwargs.get("mutation_prob", 0.2),
            "max_tree_depth": kwargs.get("max_tree_depth", 10),
            "elitism_ratio": kwargs.get("elitism_ratio", 0.05),
            "parsimony_coefficient": kwargs.get("parsimony_coefficient", 0.001),
            "multi_objective": kwargs.get("multi_objective", False),
        }

        self._build_pset(func_names)
        self._build_toolbox()
        self._setup_done = True

    def train(
        self,
        fitness_function: Callable[..., float],
        seed: int = 42,
    ) -> GPResult:
        if not self._setup_done:
            raise RuntimeError("Call setup() before train().")

        self._compiled_cache.clear()
        random.seed(seed)
        np.random.seed(seed)

        if self._params.get("multi_objective", False):
            return self._train_nsga2(fitness_function, seed)
        return self._train_single_objective(fitness_function, seed)

    def _train_single_objective(
        self,
        fitness_function: Callable[..., float],
        seed: int,
    ) -> GPResult:
        """Standard single-objective evolutionary loop."""
        toolbox = self._toolbox
        params = self._params

        pop_size = params["population_size"]
        n_gen = params["n_generations"]
        cx_prob = params["crossover_prob"]
        mut_prob = params["mutation_prob"]
        elite_count = max(1, int(pop_size * params["elitism_ratio"]))

        population = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(elite_count)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        stats.register("std", np.std)

        # Evaluate initial population
        if hasattr(fitness_function, "rotate_instances"):
            fitness_function.rotate_instances(0)
        self._evaluate_population(population, fitness_function)

        log = tools.Logbook()
        log.header = ["gen", "nevals"] + stats.fields

        record = stats.compile(population)
        log.record(gen=0, nevals=len(population), **record)
        hof.update(population)

        for gen in range(1, n_gen + 1):
            # Rotate training instances if fitness function supports it
            if hasattr(fitness_function, "rotate_instances"):
                fitness_function.rotate_instances(gen)

            # Selection
            offspring = toolbox.select(population, pop_size - elite_count)
            offspring = list(map(toolbox.clone, offspring))

            # Crossover
            for i in range(0, len(offspring) - 1, 2):
                if random.random() < cx_prob:
                    offspring[i], offspring[i + 1] = toolbox.mate(
                        offspring[i], offspring[i + 1]
                    )
                    del offspring[i].fitness.values
                    del offspring[i + 1].fitness.values

            # Mutation
            for i in range(len(offspring)):
                if random.random() < mut_prob:
                    (offspring[i],) = toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values

            # Enforce depth limit after genetic operations
            max_depth = params["max_tree_depth"]
            for ind in offspring:
                if ind.height > max_depth:
                    new_ind = toolbox.individual()
                    ind[:] = new_ind[:]
                    del ind.fitness.values

            # Evaluate individuals with invalid fitness
            invalids = [ind for ind in offspring if not ind.fitness.valid]
            self._evaluate_population(invalids, fitness_function)

            # Elitism: carry over best from previous generation
            elites = tools.selBest(population, elite_count)
            population[:] = elites + offspring

            hof.update(population)
            record = stats.compile(population)
            log.record(gen=gen, nevals=len(invalids), **record)

        best = hof[0]
        return GPResult(
            best_individual=best,
            best_fitness=best.fitness.values[0],
            best_expression=self.get_expression_string(best),
            generations=n_gen,
            log=[entry for entry in log],
            hall_of_fame=list(hof),
        )

    def _train_nsga2(
        self,
        fitness_function: Callable,
        seed: int,
    ) -> GPResult:
        """NSGA-II multi-objective evolutionary loop.

        Expects *fitness_function* to return a 3-tuple
        ``(wait_time, resource_waste, rejection_rate)`` — all minimised.
        """
        toolbox = self._toolbox
        params = self._params

        pop_size = params["population_size"]
        n_gen = params["n_generations"]
        cx_prob = params["crossover_prob"]
        mut_prob = params["mutation_prob"]
        max_depth = params["max_tree_depth"]
        coeff = params.get("parsimony_coefficient", 0.0)

        population = toolbox.population(n=pop_size)
        pareto = tools.ParetoFront()

        # Per-objective statistics
        stats = tools.MultiStatistics(
            wait=tools.Statistics(lambda ind: ind.fitness.values[0]),
            waste=tools.Statistics(lambda ind: ind.fitness.values[1]),
            reject=tools.Statistics(lambda ind: ind.fitness.values[2]),
        )
        for s in (stats["wait"], stats["waste"], stats["reject"]):
            s.register("avg", np.mean)
            s.register("min", np.min)

        # Evaluate initial population
        if hasattr(fitness_function, "rotate_instances"):
            fitness_function.rotate_instances(0)
        self._evaluate_population_mo(population, fitness_function, coeff)

        # Assign initial crowding distances via selNSGA2
        population = tools.selNSGA2(population, pop_size)

        log = tools.Logbook()
        log.header = ["gen", "nevals"]

        record = stats.compile(population)
        log.record(gen=0, nevals=len(population), **record)
        pareto.update(population)

        for gen in range(1, n_gen + 1):
            if hasattr(fitness_function, "rotate_instances"):
                fitness_function.rotate_instances(gen)

            # Parent selection: binary tournament on dominance + crowding
            # selTournamentDCD requires k divisible by 4 when k == len(pop)
            sel_k = pop_size
            if sel_k == len(population) and sel_k % 4 != 0:
                sel_k = max(4, sel_k - (sel_k % 4))
            parents = tools.selTournamentDCD(population, sel_k)
            offspring = list(map(toolbox.clone, parents))

            # Crossover
            for i in range(0, len(offspring) - 1, 2):
                if random.random() < cx_prob:
                    offspring[i], offspring[i + 1] = toolbox.mate(
                        offspring[i], offspring[i + 1]
                    )
                    del offspring[i].fitness.values
                    del offspring[i + 1].fitness.values

            # Mutation
            for i in range(len(offspring)):
                if random.random() < mut_prob:
                    (offspring[i],) = toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values

            # Enforce depth limit
            for ind in offspring:
                if ind.height > max_depth:
                    new_ind = toolbox.individual()
                    ind[:] = new_ind[:]
                    del ind.fitness.values

            # Evaluate invalidated individuals
            invalids = [ind for ind in offspring if not ind.fitness.valid]
            self._evaluate_population_mo(invalids, fitness_function, coeff)

            # Survivor selection: (μ + λ) with NSGA-II
            population = tools.selNSGA2(population + offspring, pop_size)

            pareto.update(population)
            record = stats.compile(population)
            log.record(gen=gen, nevals=len(invalids), **record)

        # Pick the "best" from the Pareto front: lowest sum of objectives
        pareto_list = list(pareto)
        best = min(pareto_list, key=lambda ind: sum(ind.fitness.values))

        return GPResult(
            best_individual=best,
            best_fitness=sum(best.fitness.values),
            best_expression=self.get_expression_string(best),
            generations=n_gen,
            log=[entry for entry in log],
            hall_of_fame=pareto_list,
            pareto_front=pareto_list,
        )

    def evaluate_individual(
        self,
        individual: Any,
        terminal_values: Dict[str, float],
    ) -> float:
        """Evaluate a GP tree on concrete terminal values.

        Uses a per-individual compiled-function cache (keyed by tree
        string representation) so that repeated evaluations during a
        single simulation run avoid recompilation overhead.
        """
        key = str(individual)
        func = self._compiled_cache.get(key)
        if func is None:
            func = gp.compile(expr=individual, pset=self._pset)
            self._compiled_cache[key] = func
        args = [terminal_values[name] for name in self._terminal_names]
        try:
            result = func(*args)
            if not np.isfinite(result):
                return 0.0
            return float(result)
        except (OverflowError, ZeroDivisionError, ValueError):
            return 0.0

    def get_expression_string(self, individual: Any) -> str:
        return str(individual)

    # ── Internal helpers ─────────────────────────────────────────────

    def _build_pset(self, func_names: List[str]) -> None:
        """Register functions and terminals with DEAP's PrimitiveSet."""
        n_terminals = len(self._terminal_names)
        pset = gp.PrimitiveSet("SCORING_RULE", arity=n_terminals)

        # Rename arguments to match terminal names
        for i, tname in enumerate(self._terminal_names):
            pset.renameArguments(**{f"ARG{i}": tname})

        # Register functions
        func_map = {
            "add": (add, 2),
            "sub": (sub, 2),
            "mul": (mul, 2),
            "protected_div": (protected_div, 2),
            "neg": (neg, 1),
            "min": (safe_min, 2),
            "max": (safe_max, 2),
            "if_positive": (if_positive, 3),
        }
        for fname in func_names:
            if fname in func_map:
                fn, arity = func_map[fname]
                pset.addPrimitive(fn, arity, name=fname)

        # Ephemeral random constant (ERC) in [-5, 5]
        def _erc_generator():
            return round(random.uniform(-5.0, 5.0), 2)

        pset.addEphemeralConstant("ERC", _erc_generator)

        self._pset = pset

    def _build_toolbox(self) -> None:
        """Configure DEAP toolbox: fitness, individual, operators."""
        multi = self._params.get("multi_objective", False)

        # Avoid duplicate creator classes during repeated setup() calls
        if multi:
            if not hasattr(creator, "FitnessMultiMin"):
                creator.create("FitnessMultiMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
            if not hasattr(creator, "IndividualMulti"):
                creator.create("IndividualMulti", gp.PrimitiveTree, fitness=creator.FitnessMultiMin)
            ind_cls = creator.IndividualMulti
        else:
            if not hasattr(creator, "FitnessMin"):
                creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            if not hasattr(creator, "Individual"):
                creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
            ind_cls = creator.Individual

        tb = base.Toolbox()
        max_depth = self._params["max_tree_depth"]

        tb.register("expr", gp.genHalfAndHalf, pset=self._pset, min_=2, max_=max_depth)
        tb.register("individual", tools.initIterate, ind_cls, tb.expr)
        tb.register("population", tools.initRepeat, list, tb.individual)
        tb.register("compile", gp.compile, pset=self._pset)

        # Selection
        tb.register(
            "select",
            tools.selTournament,
            tournsize=self._params["tournament_size"],
        )

        # Crossover (one-point)
        tb.register("mate", gp.cxOnePoint)

        # Mutation (uniform — replaces a subtree)
        tb.register(
            "mutate",
            gp.mutUniform,
            expr=functools.partial(gp.genFull, min_=1, max_=3),
            pset=self._pset,
        )

        # Bloat control via depth limit on crossover/mutation
        tb.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))
        tb.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))

        self._toolbox = tb

    def _evaluate_population(
        self,
        population: List[Any],
        fitness_function: Callable[..., float],
    ) -> None:
        """Evaluate fitness for each individual in *population*.

        Applies parsimony pressure: fitness += coefficient * tree_length.
        Since fitness is minimised, this penalises oversized trees.
        """
        coeff = self._params.get("parsimony_coefficient", 0.0)
        for ind in population:
            if not ind.fitness.valid:
                raw = fitness_function(ind)
                penalty = coeff * len(ind) if coeff > 0 else 0.0
                ind.fitness.values = (raw + penalty,)

    def _evaluate_population_mo(
        self,
        population: List[Any],
        fitness_function: Callable,
        coeff: float = 0.0,
    ) -> None:
        """Evaluate multi-objective fitness for each individual.

        *fitness_function* must return a 3-tuple (wait, waste, reject).
        Parsimony penalty is added to the first objective (wait time).
        """
        for ind in population:
            if not ind.fitness.valid:
                objs = fitness_function(ind)
                penalty = coeff * len(ind) if coeff > 0 else 0.0
                ind.fitness.values = (objs[0] + penalty, objs[1], objs[2])
