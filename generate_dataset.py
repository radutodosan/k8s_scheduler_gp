"""Trigger 1 — Generate a dynamic dataset for experiment runs.

Usage:
    py generate_dataset.py --size small    # ~1-2 min runtime
    py generate_dataset.py --size medium   # ~5 min runtime
    py generate_dataset.py --size large    # >5 min runtime
    py generate_dataset.py --size small --seed 123

The dataset is saved to ``tmp/data/dynamic_dataset/`` and will be
overwritten on each invocation.  ``main.py`` can then load it
with ``--dataset dynamic``.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List

from config.schema import ClusterConfig, ExperimentConfig, GPConfig, NodeConfig, WorkloadConfig
from models.pod import Pod
from workload.poisson_generator import PoissonWorkloadGenerator

# ── Size presets ─────────────────────────────────────────────────────
# Tuned so that a full main.py run (GP training + baselines) finishes
# within the approximate time window on a typical laptop.
# Each preset overrides a *minimal* set of ExperimentConfig defaults;
# everything else comes from the schema defaults.

PRESETS: Dict[str, Dict[str, Any]] = {
    "small": {
        "description": "Quick validation (~1-2 min)",
        "total_pods": 30,
        "nodes": 3,
        "cpu_capacity": 4.0,
        "mem_capacity": 8192.0,
        "num_training_instances": 2,
        "num_test_instances": 2,
        "gp_population_size": 30,
        "gp_n_generations": 8,
    },
    "medium": {
        "description": "Standard experiment (~5 min)",
        "total_pods": 80,
        "nodes": 5,
        "cpu_capacity": 8.0,
        "mem_capacity": 16384.0,
        "num_training_instances": 4,
        "num_test_instances": 3,
        "gp_population_size": 80,
        "gp_n_generations": 25,
    },
    "large": {
        "description": "Full-scale dissertation run (>5 min)",
        "total_pods": 200,
        "nodes": 10,
        "cpu_capacity": 8.0,
        "mem_capacity": 16384.0,
        "num_training_instances": 5,
        "num_test_instances": 5,
        "gp_population_size": 150,
        "gp_n_generations": 50,
    },
}


def preset_to_config(size: str, seed: int = 42, profile: str = "") -> ExperimentConfig:
    """Build a validated :class:`ExperimentConfig` from a size preset.

    This is the single source of truth: both ``generate_dataset`` and
    ``run_experiments`` should derive their configs from here.
    """
    preset = PRESETS[size]
    return ExperimentConfig(
        name=f"preset_{size}",
        seed=seed,
        num_training_instances=preset["num_training_instances"],
        num_test_instances=preset["num_test_instances"],
        cluster=ClusterConfig(
            node_templates=[
                NodeConfig(
                    count=preset["nodes"],
                    cpu_capacity=preset["cpu_capacity"],
                    mem_capacity=preset["mem_capacity"],
                )
            ]
        ),
        workload=WorkloadConfig(total_pods=preset["total_pods"], profile=profile),
        gp=GPConfig(
            population_size=preset["gp_population_size"],
            n_generations=preset["gp_n_generations"],
        ),
    )

DATASET_DIR = Path("tmp") / "data" / "dynamic_dataset"


# ── Serialisation helpers ────────────────────────────────────────────

def _pod_to_dict(pod: Pod) -> Dict[str, Any]:
    return {
        "pod_id": pod.pod_id,
        "cpu_request": pod.cpu_request,
        "mem_request": pod.mem_request,
        "priority": pod.priority,
        "qos_class": pod.qos_class.name,
        "arrival_time": pod.arrival_time,
        "duration": pod.duration,
        "namespace": pod.namespace,
        "tolerations": sorted(pod.tolerations),
        "node_selector": pod.node_selector,
        "cpu_limit": pod.cpu_limit,
        "mem_limit": pod.mem_limit,
        "anti_affinity_key": pod.anti_affinity_key,
        "workload_type": pod.workload_type,
        "replica_group": pod.replica_group,
    }


def _pod_from_dict(d: Dict[str, Any]) -> Pod:
    from models.pod import QoSClass
    return Pod(
        pod_id=d["pod_id"],
        cpu_request=d["cpu_request"],
        mem_request=d["mem_request"],
        priority=d["priority"],
        qos_class=QoSClass[d["qos_class"]],
        arrival_time=d["arrival_time"],
        duration=d["duration"],
        namespace=d["namespace"],
        tolerations=frozenset(d.get("tolerations", [])),
        node_selector=d.get("node_selector", {}),
        cpu_limit=d.get("cpu_limit", 0.0),
        mem_limit=d.get("mem_limit", 0.0),
        anti_affinity_key=d.get("anti_affinity_key", ""),
        workload_type=d.get("workload_type", ""),
        replica_group=d.get("replica_group", ""),
    )


def load_dataset(dataset_dir: Path = DATASET_DIR):
    """Load a previously generated dataset from *dataset_dir*.

    Returns:
        (metadata_dict, training_instances, test_instances)
    """
    meta_path = dataset_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No dataset found at {dataset_dir}. "
            "Run `py generate_dataset.py --size small` first."
        )

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    training: List[List[Pod]] = []
    for i in range(meta["num_training_instances"]):
        path = dataset_dir / "training" / f"instance_{i}.json"
        with open(path, encoding="utf-8") as f:
            training.append([_pod_from_dict(d) for d in json.load(f)])

    test: List[List[Pod]] = []
    for i in range(meta["num_test_instances"]):
        path = dataset_dir / "test" / f"instance_{i}.json"
        with open(path, encoding="utf-8") as f:
            test.append([_pod_from_dict(d) for d in json.load(f)])

    return meta, training, test


# ── Main ─────────────────────────────────────────────────────────────

def generate_dataset(size: str, seed: int = 42, profile: str = "") -> Path:
    """Generate training + test instances and persist them to disk.

    Returns the output directory path.
    """
    cfg = preset_to_config(size, seed, profile)
    preset = PRESETS[size]
    log = logging.getLogger("generate_dataset")

    # Clean previous dataset
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    training_dir = DATASET_DIR / "training"
    test_dir = DATASET_DIR / "test"
    training_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    generator = PoissonWorkloadGenerator()
    n_train = cfg.num_training_instances
    n_test = cfg.num_test_instances

    log.info("Generating %s dataset (seed=%d)", size, seed)
    log.info("  %d training instances × %d pods", n_train, cfg.workload.total_pods)
    log.info("  %d test instances × %d pods", n_test, cfg.workload.total_pods)

    for i in range(n_train):
        pods = generator.generate(cfg.workload, seed=seed + i)
        path = training_dir / f"instance_{i}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump([_pod_to_dict(p) for p in pods], f, indent=2)
        log.info("  Training instance %d: %d pods → %s", i, len(pods), path)

    for i in range(n_test):
        pods = generator.generate(cfg.workload, seed=seed + n_train + i)
        path = test_dir / f"instance_{i}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump([_pod_to_dict(p) for p in pods], f, indent=2)
        log.info("  Test instance %d: %d pods → %s", i, len(pods), path)

    # Save metadata
    node_t = cfg.cluster.node_templates[0]
    meta = {
        "size": size,
        "seed": seed,
        "profile": profile,
        "total_pods": cfg.workload.total_pods,
        "nodes": node_t.count,
        "cpu_capacity": node_t.cpu_capacity,
        "mem_capacity": node_t.mem_capacity,
        "num_training_instances": n_train,
        "num_test_instances": n_test,
        "gp_population_size": cfg.gp.population_size,
        "gp_n_generations": cfg.gp.n_generations,
    }
    with open(DATASET_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log.info("Dataset saved to %s", DATASET_DIR)
    return DATASET_DIR


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trigger 1 — Generate dynamic dataset for GP scheduler experiments",
    )
    parser.add_argument(
        "--size",
        choices=["small", "medium", "large"],
        default="small",
        help="Workload size: small (~1-2 min), medium (~5 min), large (>5 min)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--profile",
        choices=["", "web_serving", "ai_training", "ci_cd",
                 "batch_processing", "microservices", "mixed"],
        default="",
        help="Workload profile (default: generic)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    out = generate_dataset(args.size, args.seed, args.profile)
    preset = PRESETS[args.size]
    profile_str = args.profile or "generic"
    print(f"\n{'=' * 55}")
    print(f"  Dataset generated: {args.size} (profile: {profile_str})")
    print(f"  {preset['total_pods']} pods × "
          f"{preset['num_training_instances']} train + "
          f"{preset['num_test_instances']} test instances")
    print(f"  Saved to: {out}")
    print(f"{'=' * 55}")
    print(f"\nNext: py main.py --dataset dynamic --size {args.size}")
