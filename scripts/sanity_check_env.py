#!/usr/bin/env python3
"""Run a deterministic headless dynamics/control smoke test."""

import argparse

import numpy as np

from contact_deflection.envs import ContactDeflectionEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    environment = ContactDeflectionEnv()
    observation, _ = environment.reset(seed=args.seed)
    initial_projectile_velocity = environment.projectile_velocity_world
    action = np.array([0.07, -0.12, 0.06, 0.06, -0.03, 0.05])
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
