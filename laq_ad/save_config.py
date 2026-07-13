"""Write a config.json for an existing checkpoint so eval_nuscenes.py can auto-detect it.

Use this for checkpoints trained before the auto-config feature was added.

Usage:
    python save_config.py --results-folder results_laq_ad_mini --levels 8 8
    python save_config.py --results-folder results_laq_ad_mini --levels 8
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-folder", required=True, type=Path)
    p.add_argument("--levels", nargs="+", type=int, required=True,
                   help="FSQ levels used during training, e.g. --levels 8 8  or  --levels 8")
    p.add_argument("--heatmap-alpha", type=float, default=1.0)
    p.add_argument("--can-bus-weight", type=float, default=0.1)
    p.add_argument("--entropy-reg-weight", type=float, default=0.1)
    args = p.parse_args()

    cfg = {
        "levels": args.levels,
        "dim": 512,
        "image_size": [256, 256],
        "patch_size": [32, 32],
        "heatmap_alpha": args.heatmap_alpha,
        "can_bus_weight": args.can_bus_weight,
        "entropy_reg_weight": args.entropy_reg_weight,
    }
    out = args.results_folder / "config.json"
    with open(out, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Wrote {out}: {cfg}")


if __name__ == "__main__":
    main()
