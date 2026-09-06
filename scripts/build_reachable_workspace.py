"""Build/reuse constrained Mink workspace cache and export XYZ diagnostics."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from contact_deflection.control.decoder_config import StructuredDecoderConfig
from contact_deflection.envs.space_robot_env import SpaceRobotEnv
from contact_deflection.kinematics.mink_ik import MinkIK
from contact_deflection.kinematics.reachable_workspace import (
    ReachableWorkspace,
    generate_workspace,
    workspace_fingerprint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/decoder.yaml")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output")
    args = parser.parse_args()
    config = StructuredDecoderConfig.load(args.config)
    settings = config.workspace
    if args.samples is not None:
        settings = replace(settings, sample_count=args.samples)
    if args.output is not None:
        settings = replace(settings, cache_path=args.output)
    environment = SpaceRobotEnv()
    try:
        environment._reset_physics()
        # Cartesian occupancy deliberately leaves orientation uncosted.
        ik = MinkIK(environment.model, replace(config.ik, orientation_cost=0.0))
        fingerprint = workspace_fingerprint(
            ik, environment.data.qpos, settings, args.seed
        )
        path = Path(settings.cache_path)
        if path.exists():
            workspace = ReachableWorkspace.load(path, expected_fingerprint=fingerprint)
        else:
            workspace = generate_workspace(
                ik, environment.data.qpos, settings, seed=args.seed
            )
            workspace.save(path)
        workspace.export_xyz(path.with_suffix(".csv"))
        print(json.dumps(workspace.metadata, indent=2))
    finally:
        environment.close()


if __name__ == "__main__":
    main()
