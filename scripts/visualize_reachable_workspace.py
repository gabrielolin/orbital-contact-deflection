"""Render the configured smooth contact-candidate workspace."""

import argparse

from contact_deflection.control.decoder_config import StructuredDecoderConfig
from contact_deflection.envs.space_robot_env import SpaceRobotEnv
from contact_deflection.visualization import render_workspace_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/decoder.yaml")
    parser.add_argument(
        "--visualization",
        default="videos/reachable_workspace.png",
        help="PNG output path",
    )
    args = parser.parse_args()
    config = StructuredDecoderConfig.load(args.config)
    environment = SpaceRobotEnv()
    try:
        environment._reset_physics()
        output = render_workspace_snapshot(
            args.visualization,
            environment,
            config.workspace.build(),
        )
        print(f"Rendered workspace: {output}")
    finally:
        environment.close()


if __name__ == "__main__":
    main()
