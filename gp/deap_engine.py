"""DeapGeneticEngine — DEAP-based implementation of IGeneticEngine.

Provides the full GP pipeline: primitive registration, population
initialisation, evolutionary operators, and training loop.
"""

from __future__ import annotations

import functools
import logging
import operator
import random
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from deap import base, creator, gp, tools

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


logger = logging.getLogger(__name__)


def _mixed_mutate(individual: Any, pset: Any, expr_gen: Any) -> tuple:
    """Apply one of four mutations chosen randomly:
    40% mutUniform (exploratory), 30% mutNodeReplacement (surgical),
    20% mutEphemeral (numeric tuning), 10% mutShrink (anti-bloat).
    """
    r = random.random()
    if r < 0.40:
        return gp.mutUniform(individual, expr=expr_gen, pset=pset)
    elif r < 0.70:
        return gp.mutNodeReplacement(individual, pset=pset)
    elif r < 0.90:
        return gp.mutEphemeral(individual, mode="one")
    else:
        return gp.mutShrink(individual)


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
        self._seed_expressions: List[str] = []

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
            "n_restarts": kwargs.get("n_restarts", 1),
        }

        self._seed_expressions = list(kwargs.get("seed_expressions", []))
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

        n_restarts = max(1, self._params.get("n_restarts", 1))
        if n_restarts == 1:
            random.seed(seed)
            np.random.seed(seed)
            return self._train_single_objective(fitness_function, seed)

        # Multi-restart: run n_restarts independent runs, keep best by fitness
        best_result: Optional[GPResult] = None
        for i in range(n_restarts):
            restart_seed = seed + i * 97  # distinct seed per restart
            random.seed(restart_seed)
            np.random.seed(restart_seed)
            result = self._train_single_objective(fitness_function, restart_seed)
            logger.info(
                "Multi-restart %d/%d: best_fitness=%.5f",
                i + 1, n_restarts, result.best_fitness,
            )
            if best_result is None or result.best_fitness > best_result.best_fitness:
                best_result = result
        logger.info("Multi-restart winner: fitness=%.5f", best_result.best_fitness)
        return best_result

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
        self._inject_seeds(population)
        hof = tools.HallOfFame(elite_count)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)   # best quality (higher = better)
        stats.register("min", np.min)
        stats.register("std", np.std)

        # Evaluate initial population
        if hasattr(fitness_function, "rotate_instances"):
            fitness_function.rotate_instances(0)
        self._evaluate_population(population, fitness_function)

        log = tools.Logbook()
        log.header = [
            "gen", "nevals",
            "size_avg", "size_best", "depth_avg", "depth_best",
        ] + stats.fields

        record = stats.compile(population)
        record.update(self._complexity_record(population))
        log.record(gen=0, nevals=len(population), **record)
        hof.update(population)

        logger.info(
            "DEAP progress: gen=%d/%d best=%.4f avg=%.4f size_avg=%.2f size_best=%d depth_avg=%.2f depth_best=%d",
            0,
            n_gen,
            float(record["max"]),
            float(record["avg"]),
            float(record.get("size_avg", 0.0)),
            int(record.get("size_best", 0)),
            float(record.get("depth_avg", 0.0)),
            int(record.get("depth_best", 0)),
        )

        for gen in range(1, n_gen + 1):
            # Rotate training instances if fitness function supports it
            if hasattr(fitness_function, "rotate_instances"):
                fitness_function.rotate_instances(gen)

            # Fixed tournament pressure across generations.
            offspring = tools.selTournament(
                population, pop_size - elite_count, tournsize=params["tournament_size"]
            )
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

            # Diversity reinsertion: if >70% of population converged to same trees,
            # replace the worst 20% with fresh individuals (random + seeds) to escape
            # premature convergence — most common failure mode in g_webser / g_micros.
            if gen % 10 == 0:
                unique_ratio = len({str(ind) for ind in population}) / len(population)
                if unique_ratio < 0.30:
                    n_inject = max(2, int(pop_size * 0.20))
                    logger.debug(
                        "Gen %d: diversity=%.2f — reinserting %d fresh individuals",
                        gen, unique_ratio, n_inject,
                    )
                    fresh = toolbox.population(n=n_inject)
                    self._inject_seeds(fresh)
                    self._evaluate_population(fresh, fitness_function)
                    sorted_idxs = sorted(
                        range(len(population)),
                        key=lambda j: (
                            population[j].fitness.values[0]
                            if population[j].fitness.valid
                            else float("-inf")
                        ),
                    )
                    for j in range(n_inject):
                        population[sorted_idxs[j]] = fresh[j]

            hof.update(population)
            record = stats.compile(population)
            record.update(self._complexity_record(population))
            log.record(gen=gen, nevals=len(invalids), **record)

            if gen == n_gen or gen == 1 or gen % 5 == 0:
                logger.info(
                    "DEAP progress: gen=%d/%d best=%.4f avg=%.4f size_avg=%.2f size_best=%d depth_avg=%.2f depth_best=%d",
                    gen,
                    n_gen,
                    float(record["max"]),
                    float(record["avg"]),
                    float(record.get("size_avg", 0.0)),
                    int(record.get("size_best", 0)),
                    float(record.get("depth_avg", 0.0)),
                    int(record.get("depth_best", 0)),
                )

        best = hof[0]
        return GPResult(
            best_individual=best,
            best_fitness=best.fitness.values[0],
            best_expression=self.get_expression_string(best),
            generations=n_gen,
            log=[entry for entry in log],
            hall_of_fame=list(hof),
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
        """Register functions and terminals with DEAP's PrimitiveSetTyped.

        All terminals and functions operate on a single ``float`` type,
        which keeps the tree grammar simple while still enforcing type
        safety via DEAP's typed GP infrastructure.  This prevents
        invalid tree shapes (e.g. wrong arity) and makes crossover/
        mutation type-safe.
        """
        n_terminals = len(self._terminal_names)
        in_types = [float] * n_terminals
        pset = gp.PrimitiveSetTyped("SCORING_RULE", in_types=in_types, ret_type=float)

        # Rename arguments to match terminal names
        for i, tname in enumerate(self._terminal_names):
            pset.renameArguments(**{f"ARG{i}": tname})

        # Register typed functions (all float → float)
        func_map = {
            "add":           (add, [float, float], float),
            "sub":           (sub, [float, float], float),
            "mul":           (mul, [float, float], float),
            "protected_div": (protected_div, [float, float], float),
            "neg":           (neg, [float], float),
            "min":           (safe_min, [float, float], float),
            "max":           (safe_max, [float, float], float),
            "if_positive":   (if_positive, [float, float, float], float),
        }
        for fname in func_names:
            if fname in func_map:
                fn, arg_types, ret_type = func_map[fname]
                pset.addPrimitive(fn, arg_types, ret_type, name=fname)

        # Ephemeral random constant (ERC) in [-5, 5]
        def _erc_generator():
            return round(random.uniform(-5.0, 5.0), 2)

        pset.addEphemeralConstant("ERC", _erc_generator, ret_type=float)

        self._pset = pset

    def _build_toolbox(self) -> None:
        """Configure DEAP toolbox: fitness, individual, operators."""
        # Avoid duplicate creator classes during repeated setup() calls
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
        ind_cls = creator.Individual

        tb = base.Toolbox()
        max_depth = self._params["max_tree_depth"]

        # Cap initial tree height at 5 to prevent bloat explosion in gen-0 population.
        # The depth limit decorator on mate/mutate enforces max_depth during evolution.
        init_max_depth = min(max_depth, 5)
        tb.register("expr", gp.genHalfAndHalf, pset=self._pset, min_=2, max_=init_max_depth)
        tb.register("individual", tools.initIterate, ind_cls, tb.expr)
        tb.register("population", tools.initRepeat, list, tb.individual)
        tb.register("compile", gp.compile, pset=self._pset)

        # Selection
        tb.register(
            "select",
            tools.selTournament,
            tournsize=self._params["tournament_size"],
        )

        # Crossover: leaf-biased produces more varied offspring and reduces bloat
        # compared to cxOnePoint (which tends to swap large subtrees).
        tb.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)

        # Mutation: mixed strategy (see _mixed_mutate)
        expr_gen = functools.partial(gp.genFull, pset=self._pset, min_=1, max_=3)
        tb.register("mutate", _mixed_mutate, pset=self._pset, expr_gen=expr_gen)

        # Bloat control via depth limit on crossover/mutation
        tb.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))
        tb.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))

        self._toolbox = tb

    def _evaluate_population(
        self,
        population: List[Any],
        fitness_function: Callable[..., float],
    ) -> None:
        """Evaluate quality for each individual in *population*.

        Applies parsimony pressure: quality -= coefficient * tree_length.
        Since quality is maximised, subtracting the penalty discourages bloat.
        """
        coeff = self._params.get("parsimony_coefficient", 0.0)
        for ind in population:
            if not ind.fitness.valid:
                raw = fitness_function(ind)
                # Log-scaled parsimony keeps bloat under control without
                # over-penalising moderately-sized trees.
                penalty = coeff * float(np.log2(len(ind) + 1.0)) if coeff > 0 else 0.0
                ind.fitness.values = (raw - penalty,)

    @staticmethod
    def _complexity_record(population: List[Any]) -> Dict[str, float]:
        """Return average and best (by fitness sum) tree size/depth for logging."""
        if not population:
            return {
                "size_avg": 0.0,
                "size_best": 0.0,
                "depth_avg": 0.0,
                "depth_best": 0.0,
            }

        size_avg = float(sum(len(ind) for ind in population) / len(population))
        depth_avg = float(sum(ind.height for ind in population) / len(population))

        def fitness_key(ind: Any) -> float:
            vals = getattr(ind.fitness, "values", ())
            if not vals:
                return float("-inf")
            return float(sum(vals))

        best = max(population, key=fitness_key)
        return {
            "size_avg": size_avg,
            "size_best": float(len(best)),
            "depth_avg": depth_avg,
            "depth_best": float(best.height),
        }

    # ── Seeded initialization ────────────────────────────────────────────

    def _inject_seeds(self, population: List[Any]) -> None:
        """Replace a fraction of the initial population with baseline-derived seeds.

        Seeds provide GP with a head-start at the performance level of known
        good heuristics (e.g. LeastAllocated), so evolution can only improve
        from there rather than rediscovering basic rules from scratch.
        Up to 15 % of the population (minimum 1 slot) is replaced.
        Failed parses are silently skipped.
        """
        if not self._seed_expressions:
            return
        n_slots = max(1, int(len(population) * 0.15))
        ind_cls = type(population[0])
        slot = 0
        for expr in self._seed_expressions:
            if slot >= n_slots:
                break
            ind = self._parse_expression(expr, ind_cls)
            if ind is not None:
                del ind.fitness.values  # mark as unevaluated
                population[slot] = ind
                slot += 1
        if slot > 0:
            logger.debug("Seeded %d individuals from baseline expressions", slot)

    def _parse_expression(self, expr: str, ind_cls: type) -> Optional[Any]:
        """Parse a prefix-notation expression string into a DEAP individual.

        Handles terminals and functions registered in the current pset.
        Float literals (e.g. '0.93') are silently ignored — seed expressions
        should only reference named terminals.  Returns None on any error.
        """
        try:
            tokens = self._tokenize(expr)
            nodes: List[Any] = []
            pos = self._parse_tokens(tokens, 0, nodes)
            if pos != len(tokens):
                return None
            return ind_cls(nodes)
        except Exception:
            return None

    @staticmethod
    def _tokenize(expr: str) -> List[str]:
        """Split 'neg(add(X, Y))' into ['neg', 'add', 'X', 'Y'] (no parens/commas)."""
        tokens: List[str] = []
        current: List[str] = []
        for ch in expr:
            if ch in "(),":
                if current:
                    tokens.append("".join(current).strip())
                    current = []
            elif ch == " ":
                if current:
                    tokens.append("".join(current).strip())
                    current = []
            else:
                current.append(ch)
        if current:
            tokens.append("".join(current).strip())
        return [t for t in tokens if t]

    def _parse_tokens(self, tokens: List[str], pos: int, nodes: List[Any]) -> int:
        """Recursive-descent parser: appends DEAP nodes in DFS pre-order."""
        if pos >= len(tokens):
            raise ValueError("Unexpected end of token stream")
        token = tokens[pos]

        # Try functions first
        for prim in self._pset.primitives.get(float, []):
            if prim.name == token:
                nodes.append(prim)
                pos += 1
                for _ in range(prim.arity):
                    pos = self._parse_tokens(tokens, pos, nodes)
                return pos

        # Try named terminals
        for term in self._pset.terminals.get(float, []):
            if hasattr(term, "name") and term.name == token:
                nodes.append(term)
                return pos + 1

        raise ValueError(f"Unknown token {token!r} (not in pset terminals or functions)")
