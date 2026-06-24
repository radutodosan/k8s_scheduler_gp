# K8s Scheduler GP

**Genetic Programming for Dynamic Pod Scheduling in Kubernetes Clusters**

A research project that evolves scheduling rules using Genetic Programming (GP) to optimise pod placement in simulated Kubernetes clusters. Developed as part of a Master's dissertation.

---

## Overview

The default Kubernetes scheduler uses static, hand-tuned scoring rules (LeastAllocated, MostAllocated, BalancedAllocation). This project replaces those fixed heuristics with **GP-evolved scoring functions** — mathematical expressions that learn to map `(Pod, Node, ClusterState)` features to a placement score.

The system includes:

- **Discrete-event simulator** modelling a Kubernetes cluster (pods, nodes, resource accounting)
- **GP engine** (DEAP) that evolves scheduling rules as interpretable expression trees
- **Baseline strategies** (Random, RoundRobin, FirstFit, LeastAllocated, MostAllocated, BalancedAllocation, BinPacking) for comparison
- **Dynamic node failures** with three modes (off / reschedule / kill), pod eviction, restart overhead, and remaining-duration tracking
- **Taints & tolerations** — nodes can be tainted, pods must tolerate taints to be placed
- **Labels & node selectors** — pods can request specific node labels for constrained placement
- **Priority preemption** — high-priority pods can evict lower-priority ones to obtain resources
- **Requests vs limits + OOM kill** — pods have resource limits beyond requests; overcommitted nodes OOM-kill vulnerable pods
- **Pod anti-affinity** — pods with the same affinity key are spread across nodes
- **Variable arrival patterns** — workload generator supports constant, diurnal, and bursty Poisson rates
- **Workload profiles** — 6 realistic archetypes (web_serving, ai_training, ci_cd, batch_processing, microservices, mixed) with per-profile resource ranges, priorities, and scheduling parameters
- **GPU resources** — pods can request GPUs; nodes expose GPU capacity; GP terminals include `POD_GPU_REQ`, `NODE_GPU_AVAIL`, `CLUSTER_GPU_UTIL`
- **Replica groups** — pods can belong to a replica group (deployment/ReplicaSet), enabling co-location awareness
- **GP rule simplification** — algebraic simplification of evolved expression trees for interpretability (NSGA-II multi-objective available in code but not used in current experiments)
- **Cross-profile comparison** — experiment group running the same GP config across all workload profiles
- **Configuration validation** — all config dataclasses expose `validate()` methods with range checks, probability constraints, and cross-field consistency
- **Workload generator** producing realistic synthetic scenarios (Poisson arrivals, bursts, replica groups)
- **Gantt chart visualisation** of pod scheduling timelines per strategy
- **Metrics pipeline** evaluating wait time (including P50/P90/P95/P99 percentiles), resource utilisation, fairness, preemption counts, scheduling attempt counts, and rejection rates with timeline
- **Experiment framework** for systematic sweeps across engine, scale, fitness weights, GP params, dynamics, NSGA-II, and workload profiles
- **Analysis module** generating comparison tables, convergence plots, box plots, and statistical summaries
- **Web UI** — interactive configurator (YAML editor) and results dashboard served by a built-in HTTP server

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/<user>/k8s-scheduler-gp.git
cd k8s-scheduler-gp
pip install -r requirements.txt
```

### Run an experiment

```bash
# Using the default configuration
python main.py

# Using a custom config
python main.py --config config/my_experiment.yaml

# Generate Gantt charts for each strategy
python main.py --config config/my_experiment.yaml --gantt
```

### Run experiment sweeps (Chapter 5)

```bash
# Run all 19 experiments (full mode: pop=100, gen=30, seeds=5)
python run_experiments.py

# Quick validation (small parameters: pop=20, gen=5, seeds=2)
python run_experiments.py --quick

# Medium sweep (pop=80, gen=30, seeds=3)
python run_experiments.py --medium

# Overnight run (pop=120, gen=40, seeds=5, n_restarts=1, fitness_aggregation=mean_minus_std, ~6.6h)
python run_experiments.py --overnight

# Run a single experiment group
python run_experiments.py --quick --group engine

# List all available experiments
python run_experiments.py --list

# Analyse results (tables, plots, statistical report)
python analysis.py --input tmp/results/experiments
```

### Launch the Web UI

```bash
# Starts the built-in server (opens browser automatically)
python ui/server.py

# Custom port, no auto-open
python ui/server.py --port 9090 --no-browser
```

Routes:
- `http://localhost:8050/` — Results Dashboard
- `http://localhost:8050/configurator` — YAML Configurator

See [`docs/ui.md`](docs/ui.md) for detailed UI documentation.

### Run tests

```bash
pytest tests/ -v
```

**535 tests** collected by `pytest` covering: models, simulator, scheduling, GP engine (DEAP), all 7 baselines, workload generator, workload profiles, GPU resources, metrics (including wait-time percentiles, scheduling attempts, preemption tracking), config validation, dynamic instances, visualisation (Gantt, resource timelines, wait-time distributions, free resources, utilization variance, GP tree plots), node failure dynamics (off/reschedule/kill modes, restart overhead), taints/tolerations, labels/selectors, priority preemption, requests vs limits, OOM kill, anti-affinity, burst arrival patterns, replica groups, NSGA-II multi-objective, rule simplification, cross-profile comparison, experiment framework, analysis, statistical hypothesis tests, rule interpretability, resource monitoring, regression coverage, and a full integration pipeline.

## Project Structure

```
k8s_scheduler_gp/
├── main.py                     # Experiment runner (entry point)
├── run_experiments.py          # Batch experiment sweep runner (Chapter 5)
├── analysis.py                 # Results analysis: tables, plots, statistics
├── statistical.py              # Statistical tests, effect sizes, rule interpretability
├── requirements.txt
├── README.md                   # This file
├── .gitignore
│
├── ui/                         # Web UI (configurator + results dashboard)
│   ├── server.py               #   HTTP server with REST API
│   ├── configurator.html       #   YAML configuration editor
│   └── dashboard.html          #   Interactive results viewer
│
├── docs/                       # Documentation
│   ├── simulator.md            #   Simulator architecture and Kubernetes mapping
│   ├── gp.md                   #   GP theory and implementation details
│   ├── results.md              #   Metrics, experiments, analysis, statistics
│   ├── tests.md                #   Test suite documentation
│   ├── configurator.md         #   YAML schema and validation system
│   ├── ui.md                   #   Web UI documentation
│   ├── visualization.md        #   Gantt/resource/GP-tree visualizations
│   └── future_work.md          #   Future extensions and research directions
│
├── config/                     # Experiment configuration
│   ├── schema.py               #   ExperimentConfig dataclass + YAML loader + validation
│   ├── default_config.yaml     #   Default experiment parameters
│   └── smoke_test_config.yaml  #   Minimal config for fast testing
│
├── models/                     # Core domain models
│   ├── pod.py                  #   Pod, QoSClass, PodStatus
│   ├── node.py                 #   Node (capacity, allocation, utilisation)
│   └── cluster_state.py        #   ClusterState (nodes, queues, aggregates)
│
├── simulator/                  # Discrete-event simulation engine
│   ├── event.py                #   Event, EventType
│   ├── event_queue.py          #   Min-heap EventQueue
│   └── engine.py               #   SimulationEngine (main loop)
│
├── scheduling/                 # Scheduling strategies (pluggable)
│   ├── strategy.py             #   ISchedulingStrategy (abstract interface)
│   ├── gp_strategy.py          #   GPSchedulingStrategy (uses GP tree)
│   ├── random_strategy.py      #   Random baseline
│   ├── round_robin.py          #   Round-robin baseline
│   ├── first_fit.py            #   First-fit baseline
│   ├── least_allocated.py      #   Kubernetes-like least-allocated
│   ├── most_allocated.py       #   Bin-packing (fill nodes first)
│   ├── balanced_allocation.py  #   Balance CPU/memory utilisation
│   └── bin_packing.py          #   Explicit bin-packing baseline
│
├── gp/                         # Genetic Programming engines
│   ├── interface.py            #   IGeneticEngine (abstract interface)
│   ├── primitives.py           #   Terminal and function definitions
│   ├── deap_engine.py          #   DEAP-based GP engine (simulation-based fitness)
│   └── fitness.py              #   FitnessEvaluator (GP ↔ simulator bridge)
│
├── workload/                   # Synthetic workload generation
│   ├── generator.py            #   IWorkloadGenerator (abstract interface)
│   ├── profiles.py             #   Workload profiles, validation, replica groups
│   ├── poisson_generator.py    #   PoissonWorkloadGenerator (Poisson, bursts)
│   └── trace_replay.py         #   Replay support for trace-driven workloads
│
├── metrics/                    # Evaluation and reporting
│   ├── collector.py            #   MetricsCollector (per-run)
│   ├── reporter.py             #   MetricsReporter (aggregation, CSV/JSON)
│   └── resource_monitor.py     #   ResourceMonitor (time-series snapshots)
│
├── visualization/              # Output visualisation
│   ├── gantt.py                #   Gantt chart (pod timelines per node)
│   ├── resource_plots.py       #   Resource utilization time-series plots
│   └── gp_tree.py              #   GP tree and Pareto front visualisation
│
└── tests/                      # Unit and integration tests
    ├── conftest.py             #   Shared fixtures (pods, nodes, clusters, configs)
    ├── test_pod.py             #   Pod lifecycle, status transitions
    ├── test_node.py            #   Resource accounting, allocation
    ├── test_cluster_state.py   #   Cluster pod binding, feasibility
    ├── test_simulator.py       #   Event ordering, queue, SimulationEngine
    ├── test_scheduling.py      #   GPSchedulingStrategy
    ├── test_gp.py              #   Primitives, DeapEngine, FitnessEvaluator
    ├── test_workload.py        #   PoissonWorkloadGenerator determinism
    ├── test_metrics.py         #   Collector, Reporter, CSV/JSON export
    ├── test_config.py          #   YAML loading, defaults
    ├── test_gantt.py           #   Gantt chart generation and saving
    ├── test_baselines.py       #   All 7 baseline strategies + contract tests
    ├── test_experiments.py     #   Experiment framework, analysis module
    ├── test_statistical.py     #   Statistical tests, effect sizes, interpretability
    ├── test_dynamics.py        #   Node failures, eviction, recovery, metrics
    ├── test_profiles.py        #   Workload profile generation and validation
    ├── test_regression.py      #   Determinism and golden-output regression checks
    ├── test_resource_monitor.py #  ResourceMonitor, snapshots, throughput
    ├── test_resource_plots.py  #   Resource utilization plots
    └── test_integration.py     #   Full pipeline end-to-end
```

## Configuration

Experiments are configured via YAML files. See [`config/default_config.yaml`](config/default_config.yaml) for all parameters.

Key sections:

| Section | Controls |
|---------|----------|
| `cluster` | Number and size of nodes |
| `workload` | Pod arrival rates, resource distributions, burst settings, limits, anti-affinity, arrival patterns, **profile**, replica groups |
| `gp` | Population size, generations, crossover/mutation rates, tree depth, `n_workers`, `fitness_aggregation` (`"mean"` or `"mean_minus_std"`), `fitness_std_penalty`, `validation_hof_size`, `n_restarts` (run GP N times independently, keep best), optional `multi_objective` (NSGA-II, available but not used in current experiments) |
| `fitness` | Weights for wait time, resource waste, failed pods, evictions, preemptions, churn, and scheduling attempts; quality remains normalized in `[0, 1]` |
| `dynamic_instances` | When `true`, training instances are regenerated each GP generation (prevents overfitting) |
| `num_validation_instances` | Optional validation split used for DEAP hall-of-fame champion selection |
| `dynamics` | Node failure injection: `failure_mode` (off/reschedule/kill), `failure_rate` (1–3), `recovery_time_min/max`, `restart_overhead_min/max` |

All configuration dataclasses (`ClusterConfig`, `WorkloadConfig`, `GPConfig`, `FitnessWeights`, `DynamicsConfig`) expose a `validate()` method that checks value ranges, probabilities (\[0, 1\]), weight sums, cross-field consistency (e.g. `burst_size_min ≤ burst_size_max`), and enumerated choices. Call `config.validate()` before running experiments to catch errors early.

## How It Works

1. **Workload Generator** produces a set of pods with arrival times, resource requests, and priorities
2. **Simulation Engine** processes events chronologically (pod arrivals, completions, scheduling cycles)
3. At each **scheduling cycle**, the pending queue is processed: for each pod, the **scheduling strategy** scores all feasible nodes and picks the best
4. **GP-evolved rules** are expression trees that compute `Score(pod, node)` from a focused set of Kubernetes-specific terminals. 31 terminals are defined in total but only 17 are active by default (**CORE_17**: CPU/mem requests, pod wait time, pod duration, pod priority/QoS, node CPU/mem utilisation, node CPU/mem free-after, node taint count, node cost, node imbalance, pending pressure, cluster CPU/mem/GPU util, cluster health ratio). GPU-specific experiments add 3 extra terminals (CORE_20). The initial GP population is **seeded** with 11 baseline-derived expressions so evolution starts at LeastAllocated level and can only improve.
5. **GP engine (DEAP)**: simulation-based fitness — rules are directly optimised via simulation. Includes compiled-tree caching for evaluation speedup and parsimony pressure for bloat control.
6. **Fitness** is evaluated by running the simulator on multiple training instances and computing a normalised **quality score in [0, 1]** (higher = better): `quality = 1 − (α·Wₚ + β·R + γ·F + δ·E + ε·P + η·C + ζ·A)` where `Wₚ` = wait / (wait+1), `R` = `1 − mean(cpu_util, mem_util)`, `F` = rejection rate, `E` = eviction rate, `P` = preemption rate, `C` = churn rate, `A` = normalised scheduling-attempt cost
7. **Metrics Reporter** exports per-run and aggregated results to CSV/JSON, including wait-time percentiles, preemption counts, and scheduling attempt statistics

## Node Failure Dynamics

The simulator supports three failure modes via `dynamics.failure_mode`:

| Mode | Behaviour |
|------|----------|
| `off` | No failures — stable cluster (default) |
| `reschedule` | Evicted pods are re-queued with a restart overhead penalty and continue on another node |
| `kill` | Evicted pods are permanently rejected (simulates unrecoverable failures) |

**Failure scheduling**: A fixed number of failures is pre-planned based on `failure_rate` (1 = 10%, 2 = 20%, 3 = 30% of cluster nodes). Failure times are spread uniformly across [10%, 90%] of the estimated simulation duration.

- **Eviction** removes all pods from the failed node sorted by QoS class (BestEffort first, Guaranteed last), matching Kubernetes eviction semantics
- **Remaining duration** tracking ensures evicted pods resume with their remaining execution time (not full duration), preventing infinite reschedule loops
- **Restart overhead** (`restart_overhead_min/max`) adds extra time to rescheduled pods (models container image pull, init containers, health-check warm-up)
- **Recovery** restores the node after a random delay in `[recovery_time_min, recovery_time_max]`
- **GP terminal** `CLUSTER_HEALTHY_RATIO` gives the evolved rules awareness of cluster health (available/total nodes)

## Kubernetes Realism Features

The simulator models several real Kubernetes scheduling constraints beyond basic bin-packing:

### Taints & Tolerations

Nodes can carry **taints** (e.g. `gpu`, `spot`, `dedicated`). A pod can only be placed on a node if it **tolerates** all of that node's taints. Configured via `possible_taints` and `taint_toleration_probability` in `WorkloadConfig`. GP terminal: `NODE_TAINT_COUNT`.

### Labels & Node Selectors

Nodes have **labels** (e.g. `disktype: ssd`, `zone: zone-a`). Pods can specify a **node_selector** that requires matching labels. Only nodes matching all selector keys are eligible. Configured via `possible_labels` and `node_selector_probability`.

### Priority Preemption

When a high-priority pod cannot find a node with available resources, the simulator attempts **preemption**: evicting lower-priority pods (sorted by QoS then priority) from the node requiring the fewest evictions. Evicted pods are re-queued. GP terminal: `NODE_PREEMPTABLE_COUNT`.

### Requests vs Limits + OOM Kill

Pods have both `cpu_request`/`mem_request` (used for scheduling decisions) and optional `cpu_limit`/`mem_limit` (actual resource cap). When the sum of limits on a node exceeds its physical capacity, the **OOM killer** evicts non-Guaranteed pods (BestEffort first, then Burstable) until the node is no longer overcommitted. Configured via `limit_ratio_min/max` and `limit_probability`. GP terminal: `NODE_OVERCOMMIT_RATIO`.

### Pod Anti-Affinity

Pods can carry an `anti_affinity_key` (e.g. `app-web`). Two pods with the same key **cannot** be placed on the same node, forcing high-availability spread. Configured via `possible_anti_affinity_keys` and `anti_affinity_probability`. GP terminal: `NODE_AFFINITY_CONFLICT`.

### Variable Arrival Patterns

The workload generator supports three arrival modes via `arrival_pattern`:

| Mode | Behaviour |
|------|-----------|
| `constant` | Uniform Poisson rate (default) |
| `diurnal` | Sinusoidal rate — peak at hour 12, trough at hour 0 (24h cycle) |
| `bursty` | Random traffic spikes at `bursty_spike_multiplier × base_rate` |

### Workload Profiles

The generator supports **realistic workload archetypes** that model real K8s application classes. Each profile overrides resource ranges, duration, priority/QoS distributions, namespaces, and scheduling constraints:

| Profile | CPU Range | MEM Range | Duration | Arrival | Dominant QoS | Anti-Affinity |
|---------|-----------|-----------|----------|---------|-------------|---------------|
| `web_serving` | 0.1–1.0 | 128–1024 | 20–60 | diurnal | Guaranteed 50% | 0.7 |
| `ai_training` | 2.0–6.0 | 4096–16384 | 40–120 | bursty | BestEffort 50% | 0.1 |
| `ci_cd` | 0.5–2.0 | 512–4096 | 2–15 | bursty | BestEffort 60% | 0.05 |
| `batch_processing` | 1.0–4.0 | 2048–8192 | 25–80 | constant | Burstable 50% | 0.15 |
| `microservices` | 0.1–1.5 | 256–2048 | 15–50 | diurnal | Guaranteed 40% | 0.6 |
| `mixed` | *varies* | *varies* | *varies* | diurnal | *varies* | *varies* |

The **mixed** profile generates pods from all five profiles weighted by `profile_mix` (default: 30% web, 10% AI, 15% CI/CD, 20% batch, 25% microservices). Each pod carries a `workload_type` tag for post-hoc analysis.

Usage:
```yaml
workload:
  profile: "mixed"       # or "web_serving", "ai_training", etc.
  profile_mix:            # only used when profile="mixed"
    web_serving: 0.30
    ai_training: 0.10
    ci_cd: 0.15
    batch_processing: 0.20
    microservices: 0.25
```

CLI: `py generate_dataset.py --size medium --profile mixed`

## Baselines

All baselines are automatically evaluated on the test set alongside the GP-evolved rule.

| Strategy | File | Description |
|----------|------|-------------|
| Random | `scheduling/random_strategy.py` | Uniform random among feasible nodes |
| RoundRobin | `scheduling/round_robin.py` | Cyclic assignment, skips infeasible |
| FirstFit | `scheduling/first_fit.py` | First node (sorted) with capacity |
| LeastAllocated | `scheduling/least_allocated.py` | Node with most free resources (K8s default style) |
| MostAllocated | `scheduling/most_allocated.py` | Prefer fuller nodes to reduce fragmentation |
| BalancedAllocation | `scheduling/balanced_allocation.py` | Minimise CPU/memory utilisation imbalance |
| BinPacking | `scheduling/bin_packing.py` | Explicit packing-oriented baseline |
| **GP-evolved** | `scheduling/gp_strategy.py` | Learned scoring function via genetic programming |

## Resource Utilization Plots

Both `main.py` and `run_experiments.py` automatically capture per-node and cluster-level resource utilization time-series during simulation via the `ResourceMonitor`. Outputs include:

- **`resource_timelines/*.json`** — full time-series per strategy: per-node CPU/MEM utilization, per-node free resources, pod counts, node availability, cluster aggregates, CPU utilization variance, pending queue depth, completed pod count
- **`cluster_util.png`** — dual-axis cluster utilization plot (CPU + MEM fill, pending queue dashed)
- **`strategy_comparison.png`** — overlaid CPU/MEM curves across strategies when multiple monitors are available
- **`wait_time_dist.png`** — histogram of per-pod wait times with P50/P90/P99 percentile markers
- **`free_resources.png`** — dual-axis plot of free CPU (cores) and free MEM (MiB) over time
- **`util_variance.png`** — CPU utilization variance over time (load-balance indicator)
- **`gp_tree.png` / Pareto plots** — GP tree visualization and NSGA-II front projections when exported by analysis/visualization flows

These visualizations correspond to dissertation section 5.7 (*Vizualizări: Utilizare resurse în timp*).

## Gantt Charts

Pass `--gantt` to generate per-strategy Gantt charts as PNG files (saved under `<output_dir>/gantt/`).

Each chart shows:
- **Y-axis**: cluster nodes (+ an optional *Rejected* row)
- **X-axis**: simulation time
- **Bars**: pod execution spans, coloured by namespace (default) or priority
- **Lighter bars**: waiting period (arrival → scheduled)
- **Hatched bars**: rejected pods (shown on the Rejected row)

## License

Academic use — Master's dissertation project.
