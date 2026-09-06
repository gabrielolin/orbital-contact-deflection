from pathlib import Path

import numpy as np
import pytest

from contact_deflection.envs.space_robot_env import SpaceRobotEnv
from contact_deflection.kinematics.mink_ik import IKConfig, MinkIK
from contact_deflection.kinematics.reachable_workspace import (
    ReachableWorkspace,
    WorkspaceConfig,
    generate_workspace,
    workspace_fingerprint,
)


def test_workspace_cache_and_membership(tmp_path: Path) -> None:
    workspace = ReachableWorkspace(
        np.array([[0, 0, 0], [1, 0, 0]]), 0.2, {"fingerprint": "known", "seed": 7}
    )
    target = tmp_path / "workspace.npz"
    workspace.save(target)
    restored = ReachableWorkspace.load(target, expected_fingerprint="known")
    np.testing.assert_array_equal(workspace.points, restored.points)
    assert restored.metadata == workspace.metadata
    assert restored.contains(np.array([0.1, 0, 0]))
    assert not restored.contains(np.array([100, 0, 0]))
    with pytest.raises(ValueError, match="mismatch"):
        ReachableWorkspace.load(target, expected_fingerprint="other model")
    workspace.export_xyz(tmp_path / "samples.csv")
    np.testing.assert_allclose(
        np.loadtxt(tmp_path / "samples.csv", delimiter=",", skiprows=1),
        workspace.points,
    )


def test_exact_intervals_merge_overlap_preserve_gaps_and_velocity_units() -> None:
    workspace = ReachableWorkspace(np.array([[0, 0, 0], [0.2, 0, 0], [2, 0, 0]]), 0.3)
    intervals = workspace.line_intervals(np.zeros(3), np.array([2, 0, 0]))
    np.testing.assert_allclose(intervals, [(-0.15, 0.25), (0.85, 1.15)])
    assert workspace.line_intervals(np.array([0, 1, 0]), np.array([1, 0, 0])) == []
    np.testing.assert_allclose(
        workspace.line_intervals(np.zeros(3), np.array([-2, 0, 0])),
        [(-1.15, -0.85), (-0.25, 0.15)],
    )


def test_empty_stationary_and_tangent_lines() -> None:
    empty = ReachableWorkspace(np.empty((0, 3)), 0.1)
    assert not empty.contains(np.zeros(3))
    assert empty.line_intervals(np.zeros(3), np.ones(3)) == []
    workspace = ReachableWorkspace(np.zeros((1, 3)), 0.5)
    assert workspace.line_intervals(np.zeros(3), np.zeros(3)) == [(-np.inf, np.inf)]
    assert workspace.line_intervals(np.ones(3), np.zeros(3)) == []
    assert workspace.line_intervals(np.array([0, 0.5, 0]), np.array([1, 0, 0])) == [
        (0.0, 0.0)
    ]


def test_workspace_base_transform_preserves_line_parameterization() -> None:
    workspace = ReachableWorkspace(np.array([[1, 0, 0]]), 0.2)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    position = np.array([5, 2, 3])
    placed = workspace.placed(position, rotation)
    assert placed.contains(np.array([5, 3, 3]))
    np.testing.assert_allclose(
        placed.line_intervals(position, rotation @ np.array([1, 0, 0])), [(0.8, 1.2)]
    )
    with pytest.raises(ValueError, match="proper rotation"):
        workspace.placed(position, np.diag([1, 1, -1]))


def test_mink_generated_workspace_nominal_acceptance_and_provenance() -> None:
    environment = SpaceRobotEnv()
    environment._reset_physics()
    solver = MinkIK(environment.model, IKConfig(orientation_cost=0.0))
    # One deliberately unreachable uniform sample plus the nominal candidate.
    settings = WorkspaceConfig(
        sample_count=1, bounds_lower=(10, 10, 10), bounds_upper=(11, 11, 11)
    )
    initial_qpos = environment.data.qpos.copy()
    workspace = generate_workspace(solver, initial_qpos, settings, seed=3)
    nominal, _ = solver.pose(initial_qpos)
    world_workspace = workspace.placed(
        environment.spacecraft_position_world, environment.spacecraft_rotation_world
    )
    assert world_workspace.contains(nominal)
    assert not world_workspace.contains(np.full(3, 10.5))
    assert workspace.metadata["candidate_count"] == 2
    assert workspace.metadata["accepted_count"] == 1
    assert workspace.metadata["acceptance_rate"] == 0.5
    fingerprint = workspace_fingerprint(solver, initial_qpos, settings, 3)
    assert workspace.metadata["fingerprint"] == fingerprint
    assert workspace_fingerprint(solver, initial_qpos, settings, 4) != fingerprint
    np.testing.assert_array_equal(environment.data.qpos, initial_qpos)
    environment.close()
