"""Kinematic constraints and snapshot isolation on the actual robot model."""

import mujoco
import numpy as np

from contact_deflection.envs.space_robot_env import default_scene_path
from contact_deflection.kinematics.mink_ik import IKConfig, MinkIK


def test_nominal_and_nearby_goal_with_collisions_and_frozen_base() -> None:
    model = mujoco.MjModel.from_xml_path(str(default_scene_path()))
    data = mujoco.MjData(model)
    data.qpos[:3] = [0.3, -0.2, 0.1]
    original = data.qpos.copy()
    solver = MinkIK(model)
    position, rotation = solver.pose(data.qpos)
    nominal = solver.solve(data.qpos, position, rotation)
    assert nominal.converged
    nearby = solver.solve(
        data.qpos, position + [0.01, 0, 0], rotation, seed=nominal.q_target
    )
    assert nearby.converged
    assert nearby.position_residual < solver.config.position_tolerance
    assert nearby.metadata["constraints_satisfied"]
    assert nearby.metadata["minimum_collision_distance"] >= 0.002 - 1e-7
    assert np.linalg.norm(nearby.q_target - nominal.q_target) < 0.3
    assert np.all(nearby.q_target >= solver.joint_lower)
    assert np.all(nearby.q_target <= solver.joint_upper)
    np.testing.assert_array_equal(data.qpos, original)
    frozen = np.ones(model.nq, dtype=bool)
    frozen[solver.arm_qpos_indices] = False
    np.testing.assert_array_equal(solver.configuration.q[frozen], original[frozen])


def test_impossible_pose_returns_best_effort_and_joint_limits() -> None:
    model = mujoco.MjModel.from_xml_path(str(default_scene_path()))
    solver = MinkIK(
        model,
        IKConfig(max_iterations=3, joint_lower=(-0.05,) * 6, joint_upper=(0.05,) * 6),
    )
    _, orientation = solver.pose(model.qpos0)
    result = solver.solve(model.qpos0, np.array([50, 50, 50]), orientation)
    assert not result.converged
    assert result.position_residual > 10
    assert np.all(np.isfinite(result.q_target))
    assert np.all(np.abs(result.q_target) <= 0.05 + 1e-8)
    assert result.metadata["status"] != "converged"


def test_world_jacobian_matches_finite_difference() -> None:
    model = mujoco.MjModel.from_xml_path(str(default_scene_path()))
    solver = MinkIK(model)
    q = np.array([0.1, -0.3, 0.2, 0.1, -0.2, 0.3])
    jacobian = solver.jacobian(model.qpos0, q)
    epsilon = 1e-6
    for index in range(6):
        delta = np.eye(6)[index] * epsilon
        positive, _ = solver.pose(model.qpos0, q + delta)
        negative, _ = solver.pose(model.qpos0, q - delta)
        np.testing.assert_allclose(
            (positive - negative) / (2 * epsilon), jacobian[:3, index], atol=1e-7
        )


def test_collision_clearance_is_required_for_acceptance() -> None:
    model = mujoco.MjModel.from_xml_path(str(default_scene_path()))
    # Nominal wrist clearance is ~1 cm. A deliberately stricter margin means
    # matching the requested FK pose alone must not certify reachability.
    solver = MinkIK(model, IKConfig(max_iterations=1, collision_clearance=0.03))
    position, orientation = solver.pose(model.qpos0)
    result = solver.solve(model.qpos0, position, orientation)
    assert not result.converged
    assert not result.metadata["constraints_satisfied"]
