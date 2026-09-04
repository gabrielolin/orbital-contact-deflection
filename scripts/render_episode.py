#!/usr/bin/env python3
"""Render the deterministic first-milestone smoke rollout."""

import argparse

from contact_deflection.visualization import render_smoke_video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="videos/smoke.mp4")
    args = parser.parse_args()
    print(render_smoke_video(args.output, steps=args.steps, seed=args.seed))


if __name__ == "__main__":
    main()
