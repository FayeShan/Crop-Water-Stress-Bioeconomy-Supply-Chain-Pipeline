"""Load and validate the unified YAML configuration."""

import yaml
from pathlib import Path
from typing import Any, Dict, List, Tuple


class _Namespace:
    """Dot-access wrapper around a dict (one level deep)."""

    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, v)

    def __repr__(self):
        items = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"Namespace({items})"

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class HARMConfig:
    """Structured access to the YAML config."""

    def __init__(self, raw: dict, phase: str = "eval"):
        self._raw = raw
        self.phase = phase

        # Phase-specific overrides
        phase_cfg = raw.get("phases", {}).get(phase, {})

        # Paths (shared) + per-phase result_root
        p = raw["paths"]
        self.paths = _Namespace({
            "data_root":      Path(p["data_root"]),
            "pred_data_root": Path(p["pred_data_root"]),
            "result_root":    Path(phase_cfg.get("result_root", p.get("result_root", "./results"))),
            # optional paths (not always present in callers, but exposed)
            "data_root_koppen":      Path(p.get("data_root_koppen", "")) if p.get("data_root_koppen") else None,
            "pred_data_root_koppen": Path(p.get("pred_data_root_koppen", "")) if p.get("pred_data_root_koppen") else None,
            "hist_data_root":        Path(p["hist_data_root"]) if p.get("hist_data_root") else None,
            "harvest_area_dir":      Path(p["harvest_area_dir"]) if p.get("harvest_area_dir") else None,
            "spam_2020_dir":         Path(p["spam_2020_dir"]) if p.get("spam_2020_dir") else None,
            "country_shapefile":     Path(p["country_shapefile"]) if p.get("country_shapefile") else None,
            "koppen_nc":             Path(p["koppen_nc"]) if p.get("koppen_nc") else None,
        })

        self.crops: List[str] = raw["crops"]["run_list"]

        self.run = _Namespace(raw["run"])

        # Validation: start/end from phase block, but keep the other
        # validation settings (rolling windows, retrain/pred years) shared.
        val_raw = dict(raw["validation"])
        # Construct a 'fixed' sub-dict from phase-specific years
        val_raw["fixed"] = {
            "train_start": phase_cfg["train_start"],
            "train_end":   phase_cfg["train_end"],
            "eval_start":  phase_cfg["eval_start"],
            "eval_end":    phase_cfg["eval_end"],
        }
        self._validation = val_raw

        self.common = _Namespace(raw["common"])

        self.features = raw["features"]

        # Layer configs — overlay phase-specific global_tuning_method
        self.global_layer: Dict[str, Any] = dict(raw["global_layer"])
        if "global_tuning_method" in phase_cfg:
            self.global_layer["tuning_method"] = phase_cfg["global_tuning_method"]
        self.regional_layer: Dict[str, Any] = raw["regional_layer"]
        self.extreme_layer:  Dict[str, Any] = raw["extreme_layer"]

        # Evaluation — overlay phase-specific reporting flags
        self.evaluation: Dict[str, Any] = dict(raw["evaluation"])
        for k in ("include_config_comparison", "save_latex"):
            if k in phase_cfg:
                self.evaluation[k] = phase_cfg[k]

        self.plotting: Dict[str, Any] = raw["plotting"]

    @property
    def validation_mode(self) -> str:
        return self._validation["mode"]

    def validation_splits(self) -> List[Tuple[List[int], List[int]]]:
        """
        Return a list of (train_years, eval_years) tuples.

        fixed mode   -> 1 split:  [(train_years, eval_years)]
        rolling mode -> n splits: [(train_1, eval_1), (train_2, eval_2), ...]
        """
        mode = self._validation["mode"]

        if mode == "fixed":
            f = self._validation["fixed"]
            train = list(range(f["train_start"], f["train_end"] + 1))
            eval_ = list(range(f["eval_start"], f["eval_end"] + 1))
            return [(train, eval_)]

        elif mode == "rolling":
            r = self._validation["rolling"]
            splits = []
            for i in range(r["n_splits"]):
                eval_start = r["first_eval_start"] + i
                eval_end = eval_start + r["eval_window"] - 1
                train_end = eval_start - 1
                train = list(range(r["train_start"], train_end + 1))
                eval_ = list(range(eval_start, eval_end + 1))
                splits.append((train, eval_))
            return splits

        else:
            raise ValueError(f"Unknown validation mode: {mode}")

    @property
    def train_years(self) -> List[int]:
        """Train years from the first (or only) validation split."""
        return self.validation_splits()[0][0]

    @property
    def eval_years(self) -> List[int]:
        """Eval years from the first (or only) validation split."""
        return self.validation_splits()[0][1]

    @property
    def retrain_years(self) -> List[int]:
        v = self._validation
        return list(range(v["retrain_start"], v["retrain_end"] + 1))

    @property
    def pred_years(self) -> List[int]:
        v = self._validation
        return list(range(v["pred_start"], v["pred_end"] + 1))


def load_config(path: str = "config.yaml", phase: str = "eval") -> HARMConfig:
    """Load a YAML config and return a typed HARMConfig.

    phase ('eval' or 'final') selects which block under ``phases.{phase}``
    is overlaid onto the shared sections.
    """
    if phase not in ("eval", "final"):
        raise ValueError(f"phase must be 'eval' or 'final', got {phase!r}")
    with open(path) as f:
        raw = yaml.safe_load(f)
    return HARMConfig(raw, phase=phase)
