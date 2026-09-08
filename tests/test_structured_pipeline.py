"""Real-model regression from estimated line to constrained terminal solve."""

import numpy as np

from contact_deflection.control.contact_action_decoder import (
    ContactActionDecoder,
    DecodedAction,
    DecoderState,
)
from contact_deflection.control.decoder_config import TrajectoryConfig
from contact_deflection.control.trajectory import (
    JointKinematicState,
    PlaceholderTrajectoryGenerator,
)
from contact_deflection.envs.space_robot_env import SpaceRobotEnv
from contact_deflection.estimation.projectile_kf import ProjectileKalmanFilter
from contact_deflection.kinematics.mink_ik import MinkIK
from contact_deflection.kinematics.reachable_workspace import EllipsoidalWorkspace


def test_estimate_to_mink_terminal_solve_preserves_contact_request():
    env = SpaceRobotEnv()
    try:
        env._reset_physics()
        before = env.data.qpos.copy()
        ik = MinkIK(env.model)
        position, rotation = ik.pose(before)
        velocity = -rotation[:, 2] * 0.5
        estimator = ProjectileKalmanFilter(
            np.r_[position - velocity, velocity],
            np.eye(6) * 0.01,
            acceleration_noise_std=0,
            measurement_noise_std=0.01,
        )
        workspace = EllipsoidalWorkspace(position, [0.025, 0.025, 0.025])
        corridor = estimator.intercept_corridor(workspace)
        decoder = ContactActionDecoder(
            ik, PlaceholderTrajectoryGenerator(TrajectoryConfig().limits(ik))
        )
        state = DecoderState(before, JointKinematicState(env.arm_q, env.arm_qd), 0.0)
        result = decoder.decode(state, corridor, np.zeros(8))
        assert isinstance(result, DecodedAction)
        goal = result.requested_contact_goal
        np.testing.assert_allclose(goal.position_W, position)
        assert np.isclose(goal.contact_time, 1.0)
        np.testing.assert_allclose(
            goal.position_W, estimator.spacetime_line().evaluate(goal.line_parameter)[0]
        )
        assert result.ik_solution.converged
        assert result.ik_solution.metadata["constraints_satisfied"]
        assert result.trajectory_result.status == "not_implemented"
        assert result.trajectory_result.trajectory is None
        np.testing.assert_array_equal(env.data.qpos, before)
    finally:
        env.close()
