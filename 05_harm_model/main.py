#!/usr/bin/env python3
"""HARM pipeline main CLI: dispatches sub-commands to the train/predict/evaluate scripts under src/, forwarding args (including --config and --phase). Run with -h to list sub-commands."""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# Directory containing this main.py (and the ``harm/`` package).
ROOT = Path(__file__).resolve().parent


# The weighted variants of the Extreme layer and the final retrain are
# reached by passing ``--weighted`` to ``train-extreme`` and ``retrain``
# respectively; there is no separate sub-command for them.
COMMANDS = {
    "train":          "src/train/run_pipeline.py",
    "train-global":   "src/train/run_global.py",
    "train-regional": "src/train/run_regional.py",
    "train-extreme":  "src/train/run_extreme.py",   # +--weighted for GW+RW base
    "retrain":        "src/train/run_retrain.py",   # +--weighted for GW+RW+E cascade
    "predict":        "src/predict/run_predict.py",
    "eval-predict":   "src/predict/run_eval_predictions.py",
    "export-predictions": "src/predict/export_predictions.py",  # parquet -> flat {crop}.csv for Stage 06/08
    "multilevel":     "src/evaluate/compute_multilevel.py",
    "country-eval":   "src/evaluate/country_eval.py",
    "metric-tables":  "src/evaluate/build_metric_tables.py",
    "koppen-regions": "src/utils/replace_regiontype_koppen.py",
}

# Sub-commands that touch wh/mai training parquets
_TRAIN_CMDS = {"train", "train-global", "train-regional", "train-extreme", "retrain"}
_PREDICT_CMDS = {"predict", "eval-predict"}


def print_help():
    print(__doc__)


def _extract_value(argv: List[str], flag: str, default: Optional[str] = None) -> Optional[str]:
    """Extract the value of ``--flag`` or ``--flag=VAL`` from argv; returns default otherwise."""
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return default


def _extract_list(argv: List[str], flag: str) -> List[str]:
    """Extract the list of values following ``--flag`` (stops at next --option)."""
    out: List[str] = []
    try:
        i = argv.index(flag)
    except ValueError:
        return out
    for a in argv[i + 1:]:
        if a.startswith("--"):
            break
        out.append(a)
    return out


def _preflight(cmd: str, forward: List[str]) -> None:
    """Run any required pre-flight checks before dispatching."""
    if cmd not in (_TRAIN_CMDS | _PREDICT_CMDS):
        return

    crops = _extract_list(forward, "--crops")
    if not crops:
        # If the user did not pass --crops, we cannot tell which crops
        # will run; the training scripts fall back to config.crops.run_list
        # in that case. We still check, using the config value.
        from harm.config import load_config
        config_path = _extract_value(forward, "--config", default="config.yaml")
        phase = _extract_value(forward, "--phase", default="eval")
        try:
            cfg = load_config(config_path, phase=phase)
            crops = cfg.crops
        except Exception:
            return  # loader will raise a clearer error downstream

    from harm.preflight import check_koppen_data_available
    mode = "train" if cmd in _TRAIN_CMDS else "predict"
    config_path = _extract_value(forward, "--config", default="config.yaml")
    phase = _extract_value(forward, "--phase", default="eval")
    check_koppen_data_available(crops, config_path=config_path, phase=phase, mode=mode)


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print_help()
        return 0

    cmd = argv[0]
    if cmd not in COMMANDS:
        print(f"Unknown sub-command: {cmd!r}\n", file=sys.stderr)
        print_help()
        return 2

    forward = argv[1:]

    # Pre-flight checks (raises SystemExit with a helpful message on failure)
    _preflight(cmd, forward)

    # Delegate to the script with the remaining args. The ``harm``
    # package lives at the project root, so we inject that directory
    # onto ``PYTHONPATH`` before invoking the sub-script.
    script = COMMANDS[cmd]
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + existing if existing else "")
    return subprocess.call([sys.executable, script] + forward, env=env)


if __name__ == "__main__":
    sys.exit(main())
