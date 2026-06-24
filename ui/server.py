"""Lightweight HTTP server for the K8s GP Scheduler UI.

Serves the configurator and results dashboard, plus a REST-like
JSON API that reads result files from tmp/.

Usage:
    py ui/server.py                   # http://localhost:8050
    py ui/server.py --port 9090       # custom port
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import logging
import os
import re
import webbrowser
import xml.etree.ElementTree as ET
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)

UI_DIR = Path(__file__).parent          # ui/
ROOT = UI_DIR.parent                    # project root (k8s_scheduler_gp/)
TMP = ROOT / "tmp"


# ═══════════════════════════════════════════════════════════════════════
# Data readers
# ═══════════════════════════════════════════════════════════════════════

def _list_test_runs() -> List[Dict[str, Any]]:
    """Return metadata for each pytest run in tmp/tests_results/."""
    results = []
    tests_dir = TMP / "tests_results"
    if not tests_dir.exists():
        return results
    for d in sorted(tests_dir.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        xml_path = d / "results.xml"
        info: Dict[str, Any] = {"name": d.name, "path": str(d.relative_to(ROOT))}
        if xml_path.exists():
            try:
                tree = ET.parse(xml_path)
                ts = tree.find(".//testsuite")
                if ts is not None:
                    info["tests"] = int(ts.get("tests", 0))
                    info["failures"] = int(ts.get("failures", 0))
                    info["errors"] = int(ts.get("errors", 0))
                    info["skipped"] = int(ts.get("skipped", 0))
                    info["time"] = float(ts.get("time", 0))
                    info["timestamp"] = ts.get("timestamp", "")
                    # parse individual test cases
                    cases = []
                    for tc in tree.iter("testcase"):
                        case: Dict[str, Any] = {
                            "classname": tc.get("classname", ""),
                            "name": tc.get("name", ""),
                            "time": float(tc.get("time", 0)),
                            "status": "passed",
                        }
                        if tc.find("failure") is not None:
                            case["status"] = "failed"
                            case["message"] = tc.find("failure").get("message", "")
                        elif tc.find("error") is not None:
                            case["status"] = "error"
                        elif tc.find("skipped") is not None:
                            case["status"] = "skipped"
                        cases.append(case)
                    info["cases"] = cases
            except Exception:
                pass
        results.append(info)
    return results


def _list_single_runs() -> List[Dict[str, Any]]:
    """Return metadata for each single run in tmp/results/runs/."""
    results = []
    runs_dir = TMP / "results" / "runs"
    if not runs_dir.exists():
        return results
    for d in sorted(runs_dir.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        info: Dict[str, Any] = {"name": d.name, "path": str(d.relative_to(ROOT))}

        # Read CSV — try well-known name first, then any *_results.csv in the run dir
        csv_path = d / "default_experiment_results.csv"
        if not csv_path.exists():
            candidates = sorted(d.glob("*_results.csv"))
            if candidates:
                csv_path = candidates[0]
        if csv_path.exists():
            info["csv"] = _read_csv(csv_path)

        # Read GP rule
        rule_path = d / "gp_evolved_rule.txt"
        if rule_path.exists():
            info["rule"] = rule_path.read_text(encoding="utf-8")

        # Read per-engine GP rule files (gp_evolved_rule_deap.txt etc.)
        rules: Dict[str, str] = {}
        for p in sorted(d.glob("gp_evolved_rule_*.txt")):
            engine_name = p.stem.replace("gp_evolved_rule_", "")
            if engine_name:
                rules[engine_name] = p.read_text(encoding="utf-8")
        if rules:
            info["rules"] = rules

        # List all PNG plots generated for this run (timeline + visualizations + any future folders)
        info["images"] = [
            str(f.relative_to(ROOT))
            for f in sorted(d.rglob("*.png"))
            if f.is_file()
        ]

        # Timeline JSONs remain grouped from resource_timelines/ for timeline-specific consumers
        timeline_dir = d / "resource_timelines"
        if timeline_dir.exists():
            info["timelines"] = [
                str(f.relative_to(ROOT))
                for f in sorted(timeline_dir.iterdir())
                if f.suffix == ".json"
            ]

        results.append(info)
    return results


def _list_experiment_sweeps() -> List[Dict[str, Any]]:
    """Return metadata for each experiment sweep in tmp/results/experiments/."""
    results = []
    exp_dir = TMP / "results" / "experiments"
    if not exp_dir.exists():
        return results
    for d in sorted(exp_dir.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        info: Dict[str, Any] = {"name": d.name, "path": str(d.relative_to(ROOT))}

        # Parse directory name — formats supported:
        #   current: "2026.06.02_00-48-03_medium"  → dots in date, dashes in time
        #   old1:    "2026_06_02_0000_medium_results"
        #   old2:    "2026_03_29_2011_results"
        known_presets = {"quick", "medium", "day", "overnight", "full"}
        name_clean = d.name.replace("_results", "")
        parts = name_clean.split("_")
        preset = next((p for p in parts if p in known_presets), "unknown")

        run_ts = name_clean  # fallback
        import re as _re
        # New format: "2026_06_02_00_48_03_medium" (YYYY_MM_DD_HH_MM_SS_preset)
        m = _re.match(r'^(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})', name_clean)
        if m:
            run_ts = f"{m.group(1)}.{m.group(2)}.{m.group(3)}_{m.group(4)}:{m.group(5)}:{m.group(6)}"
        else:
            # Old format: "2026_06_02_0048" (YYYY_MM_DD_HHMM)
            m2 = _re.match(r'^(\d{4})_(\d{2})_(\d{2})_(\d{2})(\d{2})', name_clean)
            if m2:
                run_ts = f"{m2.group(1)}.{m2.group(2)}.{m2.group(3)}_{m2.group(4)}:{m2.group(5)}"

        info["run_ts"] = run_ts
        info["preset"] = preset

        # ── Multi-seed support ───────────────────────────────────────
        seed_dirs = sorted(
            [sub for sub in d.iterdir() if sub.is_dir() and sub.name.startswith("seed_")],
            key=lambda p: int(p.name.split("_")[1]) if p.name.split("_")[1].isdigit() else 0,
        )
        info["is_multiseed"] = bool(seed_dirs)
        info["seeds"] = [sd.name for sd in seed_dirs]

        ms_csv = d / "multiseed_results.csv"
        if ms_csv.exists():
            info["multiseed_csv"] = _read_csv(ms_csv)

        ms_summary = d / "multiseed_summary.txt"
        if ms_summary.exists():
            info["multiseed_summary"] = ms_summary.read_text(encoding="utf-8")

        # Acceptance criteria report (generated by analysis.py --multiseed)
        acc_path = d / "analysis" / "acceptance_criteria.txt"
        if acc_path.exists():
            info["acceptance_criteria"] = acc_path.read_text(encoding="utf-8")

        # ── Single-seed combined CSV (legacy / non-multiseed runs) ───
        combined = d / "combined_results.csv"
        if combined.exists():
            info["combined_csv"] = _read_csv(combined)

        # Summary text
        summary = d / "experiment_summary.txt"
        if summary.exists():
            info["summary"] = summary.read_text(encoding="utf-8")

        # ── Per-experiment metadata/convergence ──────────────────────
        # Aggregate rules from ALL seeds (not just the first).
        # Also read convergence from first seed for per-card display.
        exp_source_dir = seed_dirs[0] if seed_dirs else d

        # Build all_rules: {exp_name: [{seed, expression, fitness, training_time_s}]}
        all_rules: Dict[str, List[Dict[str, Any]]] = {}
        for sd in seed_dirs:
            for exp_sub in sorted(sd.iterdir()):
                if not exp_sub.is_dir():
                    continue
                rule_path = exp_sub / "gp_rule.json"
                if rule_path.exists():
                    with open(rule_path, encoding="utf-8") as f:
                        rule_data = json.load(f)
                    all_rules.setdefault(exp_sub.name, []).append(rule_data)

        experiments = []
        all_source_dirs = seed_dirs if seed_dirs else [exp_source_dir]
        exp_names = sorted({
            sub.name
            for sd in all_source_dirs
            for sub in sd.iterdir()
            if sub.is_dir()
        })
        for exp_name in exp_names:
            first_sub = exp_source_dir / exp_name
            exp_info: Dict[str, Any] = {"name": exp_name}

            # GP rule + metadata from first seed (representative)
            rule_path = first_sub / "gp_rule.json"
            if rule_path.exists():
                with open(rule_path, encoding="utf-8") as f:
                    exp_info["gp_rule"] = json.load(f)
            else:
                meta_path = first_sub / "metadata.json"
                if meta_path.exists():
                    with open(meta_path, encoding="utf-8") as f:
                        exp_info["metadata"] = json.load(f)

            # All rules from every seed (for rules explorer)
            if exp_name in all_rules:
                exp_info["all_rules"] = sorted(
                    all_rules[exp_name],
                    key=lambda r: r.get("best_fitness", 0),
                    reverse=True,
                )

            # Convergence from first seed (learning curve)
            conv_path = first_sub / "convergence.json"
            if conv_path.exists():
                with open(conv_path, encoding="utf-8") as f:
                    exp_info["convergence"] = json.load(f)

            # Config from first seed (same structure across seeds)
            cfg_path = first_sub / "experiment_config.json"
            if cfg_path.exists():
                with open(cfg_path, encoding="utf-8") as f:
                    exp_info["experiment_config"] = json.load(f)

            # Results aggregated from ALL available seeds
            all_seed_results: List[Dict[str, Any]] = []
            for sd in all_source_dirs:
                csv_path = sd / exp_name / "results.csv"
                if csv_path.exists():
                    rows = _read_csv(csv_path)
                    for row in rows:
                        row["run_seed"] = sd.name   # always overwrite (CSV has no this column)
                    all_seed_results.extend(rows)
            if all_seed_results:
                exp_info["results"] = all_seed_results
                # Cross-seed mean quality per strategy (sorted best→worst)
                from collections import defaultdict as _dd
                strat_qs: Dict[str, List[float]] = _dd(list)
                for row in all_seed_results:
                    try:
                        strat_qs[row["strategy"]].append(float(row["quality_score"]))
                    except (KeyError, ValueError):
                        pass
                def _make_summary(rows_subset):
                    sq: Dict[str, List[float]] = _dd(list)
                    for row in rows_subset:
                        try:
                            sq[row["strategy"]].append(float(row["quality_score"]))
                        except (KeyError, ValueError):
                            pass
                    def _median(vals):
                        s = sorted(vals)
                        m = len(s) // 2
                        return round((s[m] if len(s) % 2 else (s[m-1] + s[m]) / 2), 4)
                    return [
                        {
                            "strategy": s,
                            "mean": round(sum(v) / len(v), 4),
                            "median": _median(v),
                            "std": round((sum((x - sum(v)/len(v))**2 for x in v) / len(v)) ** 0.5, 4),
                            "n": len(v),
                        }
                        for s, v in sorted(sq.items(), key=lambda kv: -sum(kv[1])/len(kv[1]) if kv[1] else 0)
                        if v
                    ]

                exp_info["results_summary"] = _make_summary(all_seed_results)

                # Per-seed summary (same structure, split by seed)
                exp_info["results_summary_per_seed"] = [
                    {
                        "seed": sd.name,
                        "rows": _make_summary([r for r in all_seed_results if r.get("run_seed") == sd.name]),
                    }
                    for sd in all_source_dirs
                    if any(r.get("run_seed") == sd.name for r in all_seed_results)
                ]

            experiments.append(exp_info)
        info["experiments"] = experiments

        # ── Analysis outputs ─────────────────────────────────────────
        analysis_dir = d / "analysis"
        if analysis_dir.exists():
            info["analysis"] = {}
            for txt in (
                "analysis_report.txt", "statistical_report.txt",
                "multiseed_statistics.csv",
            ):
                p = analysis_dir / txt
                if p.exists():
                    info["analysis"][txt] = (
                        _read_csv(p) if txt.endswith(".csv")
                        else p.read_text(encoding="utf-8")
                    )
            plots_dir = analysis_dir / "plots"
            if plots_dir.exists():
                info["analysis"]["plots"] = [
                    str(f.relative_to(ROOT))
                    for f in sorted(plots_dir.iterdir())
                    if f.suffix == ".png"
                ]
            # All PNGs in analysis dir (includes convergence, box plots, terminal freq, verdicts)
            all_analysis_plots = sorted(
                [f for f in analysis_dir.iterdir() if f.suffix == ".png"],
                key=lambda f: f.name,
            )
            if all_analysis_plots:
                # Categorise by prefix
                info["analysis"]["convergence_plots"] = [
                    str(f.relative_to(ROOT)) for f in all_analysis_plots
                    if f.name.startswith("convergence_")
                ]
                info["analysis"]["box_plots"] = [
                    str(f.relative_to(ROOT)) for f in all_analysis_plots
                    if f.name.startswith("box_multiseed_")
                ]
                info["analysis"]["terminal_frequency_plot"] = next(
                    (str(f.relative_to(ROOT)) for f in all_analysis_plots
                     if f.name == "terminal_frequency.png"), None
                )
                info["analysis"]["verdicts_plot"] = next(
                    (str(f.relative_to(ROOT)) for f in all_analysis_plots
                     if f.name == "verdicts_summary.png"), None
                )
                info["analysis"]["multiseed_plots"] = [
                    str(f.relative_to(ROOT)) for f in all_analysis_plots
                ]

        results.append(info)
    return results


def _read_csv(path: Path) -> List[Dict[str, str]]:
    """Read a CSV into a list of dicts."""
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _serve_image_base64(rel_path: str) -> str | None:
    """Read the image at rel_path and return base64-encoded data URL."""
    full = ROOT / rel_path
    if not full.exists() or not full.is_file():
        return None
    # Validate it's within ROOT to prevent directory traversal
    try:
        full.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return None
    suffix = full.suffix.lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "gif": "image/gif", "svg": "image/svg+xml"}.get(suffix.lstrip("."), "image/png")
    data = full.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _read_timeline_json(rel_path: str) -> Any:
    """Read a timeline JSON file."""
    full = ROOT / rel_path
    if not full.exists():
        return None
    try:
        full.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return None
    with open(full, encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════
# HTTP Handler
# ═══════════════════════════════════════════════════════════════════════

class UIHandler(SimpleHTTPRequestHandler):
    """Serves configurator + dashboard HTML and the JSON results API."""

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/":
            self._serve_file(UI_DIR / "dashboard.html", "text/html")
        elif path == "/dashboard":
            self._serve_file(UI_DIR / "dashboard.html", "text/html")
        elif path == "/configurator":
            self._serve_file(UI_DIR / "configurator.html", "text/html")
        elif path == "/shared.js":
            self._serve_file(UI_DIR / "shared.js", "application/javascript")
        elif path == "/api/tests":
            self._json_response(_list_test_runs())
        elif path == "/api/runs":
            self._json_response(_list_single_runs())
        elif path == "/api/experiments":
            self._json_response(_list_experiment_sweeps())
        elif path == "/api/image":
            p = qs.get("path", [None])[0]
            if p:
                # Sanitize: only allow paths under tmp/
                p = p.replace("\\", "/")
                if not p.startswith("tmp/"):
                    self._json_response({"error": "forbidden"}, 403)
                    return
                data_url = _serve_image_base64(p)
                if data_url:
                    self._json_response({"data_url": data_url})
                else:
                    self._json_response({"error": "not found"}, 404)
            else:
                self._json_response({"error": "missing path"}, 400)
        elif path == "/api/timeline":
            p = qs.get("path", [None])[0]
            if p:
                p = p.replace("\\", "/")
                if not p.startswith("tmp/"):
                    self._json_response({"error": "forbidden"}, 403)
                    return
                data = _read_timeline_json(p)
                if data is not None:
                    self._json_response(data)
                else:
                    self._json_response({"error": "not found"}, 404)
            else:
                self._json_response({"error": "missing path"}, 400)
        else:
            self.send_error(404)

    def _serve_file(self, fpath: Path, content_type: str):
        if not fpath.exists():
            self.send_error(404)
            return
        data = fpath.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json_response(self, obj: Any, status: int = 200):
        data = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        logger.debug(fmt, *args)


def main():
    parser = argparse.ArgumentParser(description="K8s GP Scheduler UI Server")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    server = HTTPServer(("127.0.0.1", args.port), UIHandler)
    url = f"http://localhost:{args.port}"
    print(f"K8s GP Scheduler UI: {url}")
    print(f"  Dashboard:    {url}/dashboard")
    print(f"  Configurator: {url}/configurator")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutdown.")
        server.server_close()


if __name__ == "__main__":
    main()
