"""Render the configured smooth contact-candidate workspace."""

import argparse

from contact_deflection.control.decoder_config import StructuredDecoderConfig
from contact_deflection.envs.space_robot_env import SpaceRobotEnv
from contact_deflection.visualization import (
    render_workspace_snapshot,
    view_workspace_live,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/decoder.yaml")
    parser.add_argument(
        "--visualization",
        default="videos/reachable_workspace.png",
        help="PNG output path",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="open an interactive MuJoCo viewer instead of writing a PNG",
    )
    args = parser.parse_args()
    config = StructuredDecoderConfig.load(args.config)
    environment = SpaceRobotEnv()
    try:
        environment._reset_physics()
        workspace = config.workspace.build()
        if args.live:
            print("Interactive workspace viewer open; close the window to exit.")
            view_workspace_live(environment, workspace)
        else:
            output = render_workspace_snapshot(
                args.visualization,
                environment,
                workspace,
            )
            print(f"Rendered workspace: {output}")
    finally:
        environment.close()


if __name__ == "__main__":
    main()
