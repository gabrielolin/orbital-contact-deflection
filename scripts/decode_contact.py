"""Inspect a Box(8) decode without executing or training a policy."""

import argparse
import json

import numpy as np

from contact_deflection.control.contact_action_decoder import (
    ContactActionDecoder,
    DecoderState,
)
from contact_deflection.control.decoder_config import StructuredDecoderConfig
from contact_deflection.control.trajectory import (
    JointKinematicState,
    PlaceholderTrajectoryGenerator,
)
from contact_deflection.envs import ContactDeflectionEnv
from contact_deflection.estimation.projectile_kf import ProjectileKalmanFilter
from contact_deflection.estimation.spacetime_line import (
    NoReachableWorkspaceIntersection,
)
from contact_deflection.kinematics.mink_ik import MinkIK
from contact_deflection.kinematics.reachable_workspace import ReachableWorkspace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/decoder.yaml")
    parser.add_argument("--workspace")
    parser.add_argument("--action", type=float, nargs=8, default=[0.0] * 8)
    parser.add_argument(
        "--nominal-line",
        action="store_true",
        help="Use a synthetic mean path through the nominal shield center",
    )
    args = parser.parse_args()
    config = StructuredDecoderConfig.load(args.config)
    env = ContactDeflectionEnv()
    try:
        env.reset(seed=0)
        ik = MinkIK(env.model, config.ik)
        workspace = ReachableWorkspace.load(
            args.workspace or config.workspace.cache_path
        )
        world_workspace = workspace.placed(
            env.spacecraft_position_world, env.spacecraft_rotation_world
        )
        if args.nominal_line:
            position, orientation = ik.pose(env.data.qpos)
            velocity = -orientation[:, 2] * 0.5
            estimator = ProjectileKalmanFilter(
                np.r_[position - velocity, velocity],
                np.eye(6) * 0.01,
                acceleration_noise_std=0,
                measurement_noise_std=0.01,
            )
            corridor = estimator.intercept_corridor(
                world_workspace, horizon=config.prediction_horizon
            )
        else:
            corridor = env.projectile_intercept_corridor(
                world_workspace, horizon=config.prediction_horizon
            )
        decoder = ContactActionDecoder(
            ik,
            PlaceholderTrajectoryGenerator(config.trajectory.limits(ik)),
            config.decoder,
        )
        state = DecoderState(
            env.data.qpos.copy(),
            JointKinematicState(env.arm_q, env.arm_qd),
            env.data.time,
        )
        result = decoder.decode(state, corridor, np.asarray(args.action))
        if isinstance(result, NoReachableWorkspaceIntersection):
            print(json.dumps({"status": result.reason}))
        else:
            goal = result.requested_contact_goal
            print(
                json.dumps(
                    {
                        "position_W": goal.position_W.tolist(),
                        "contact_time": goal.contact_time,
                        "ik_converged": result.ik_solution.converged,
                        "ik_position_residual": result.ik_solution.position_residual,
                        "twist_residual": result.twist_residual.tolist(),
                        "trajectory_status": result.trajectory_result.status,
                    },
                    indent=2,
                )
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
