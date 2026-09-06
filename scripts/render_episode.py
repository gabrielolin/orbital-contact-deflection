#!/usr/bin/env python3
"""Render the deterministic first-milestone smoke rollout."""

import argparse

from contact_deflection.control.decoder_config import StructuredDecoderConfig
from contact_deflection.envs.contact_deflection_env import ContactDeflectionConfig
from contact_deflection.rl.train import ensure_workspace
from contact_deflection.visualization import render_smoke_video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="videos/smoke.mp4")
    parser.add_argument("--decoder-config", default="configs/decoder.yaml")
    parser.add_argument("--workspace", default=None)
    args = parser.parse_args()
    decoder = StructuredDecoderConfig.load(args.decoder_config)
    task = ContactDeflectionConfig(decoder=decoder)
    workspace = ensure_workspace(
        task, args.workspace or decoder.workspace.cache_path, seed=args.seed
    )
    print(
        render_smoke_video(
            args.output, workspace, task, steps=args.steps, seed=args.seed
        )
    )


if __name__ == "__main__":
    main()
