#!/usr/bin/env python3
"""Run a deterministic headless dynamics/control smoke test."""

import argparse

import numpy as np

from contact_deflection.control.decoder_config import StructuredDecoderConfig
from contact_deflection.envs import ContactDeflectionEnv
from contact_deflection.envs.contact_deflection_env import ContactDeflectionConfig
from contact_deflection.rl.train import ensure_workspace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decoder-config", default="configs/decoder.yaml")
    parser.add_argument("--workspace", default=None)
    args = parser.parse_args()
    decoder = StructuredDecoderConfig.load(args.decoder_config)
    task = ContactDeflectionConfig(decoder=decoder)
    workspace = ensure_workspace(
        task, args.workspace or decoder.workspace.cache_path, seed=args.seed
    )
    environment = ContactDeflectionEnv(workspace, task)
    observation, _ = environment.reset(seed=args.seed)
    initial_projectile_velocity = environment.projectile_velocity_world
    action = np.zeros(8)
    info = {}
    for _ in range(args.steps):
        observation, _, terminated, truncated, info = environment.step(action)
        if terminated or truncated:
            break
    values = np.concatenate([environment.data.qpos, environment.data.qvel, observation])
    if not np.all(np.isfinite(values)):
        raise RuntimeError("non-finite MuJoCo state")
    print(f"requested_steps={args.steps} time={environment.data.time:.6f}")
    print(f"projectile_position={environment.projectile_position_world.tolist()}")
    velocity_change = np.linalg.norm(
        environment.projectile_velocity_world - initial_projectile_velocity
    )
    print(f"projectile_velocity_change={velocity_change:.9f}")
    print(f"spacecraft_displacement={info['base_displacement']:.9f}")
    environment.close()


if __name__ == "__main__":
    main()
