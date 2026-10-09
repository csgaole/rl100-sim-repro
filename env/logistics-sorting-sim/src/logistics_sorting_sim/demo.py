"""Hold-position smoke demo; deliberately not a successful sorting controller."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .task import LogisticsSortingEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sku", choices=["SKU02", "SKU09"], default="SKU09")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("demo_output"))
    args = parser.parse_args()
    with LogisticsSortingEnv(sku=args.sku, render=args.render) as env:
        obs, _ = env.reset(seed=args.seed)
        if args.render:
            args.output.mkdir(parents=True, exist_ok=True)
            for camera, rgb in obs["rgb"].items():
                Image.fromarray(rgb).save(args.output/f"{camera}.png")
        action = obs["proprio"][:14].copy()
        while True:
            _, _, terminated, truncated, info = env.step(np.tile(action, (8, 1)))
            if terminated or truncated:
                break
        print(json.dumps({"sku": args.sku, "seed": args.seed, "reason": info["reason"],
                          "sim_seconds": float(env.physics.data.time),
                          "controller": "hold_position_not_an_expert"}, indent=2))


if __name__ == "__main__":
    main()
