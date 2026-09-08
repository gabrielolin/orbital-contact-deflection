import numpy as np
import pytest

from contact_deflection.kinematics.reachable_workspace import (
    EllipsoidalWorkspace,
    WorkspaceConfig,
)


def test_config_builds_smooth_workspace_and_validates_radii() -> None:
    config = WorkspaceConfig(center=(1, 2, 3), radii=(0.5, 1.0, 2.0))
    workspace = config.build()
    np.testing.assert_allclose(workspace.center, [1, 2, 3])
    np.testing.assert_allclose(workspace.radii, [0.5, 1.0, 2.0])
    assert workspace.contains([1.25, 2, 3])
    assert not workspace.contains([1.51, 2, 3])
    with pytest.raises(ValueError, match="positive"):
        WorkspaceConfig(radii=(1, 0, 1))


def test_ellipsoid_line_intersection_preserves_velocity_units() -> None:
    workspace = EllipsoidalWorkspace([3, 0, 0], [0.5, 1, 2])
    np.testing.assert_allclose(
        workspace.line_intervals([1, 0, 0], [2, 0, 0]), [(0.75, 1.25)]
    )
    np.testing.assert_allclose(
        workspace.line_intervals([5, 0, 0], [-2, 0, 0]), [(0.75, 1.25)]
    )
    assert workspace.line_intervals([0, 2, 0], [1, 0, 0]) == []


def test_stationary_and_tangent_lines() -> None:
    workspace = EllipsoidalWorkspace([0, 0, 0], [0.5, 1, 2])
    assert workspace.line_intervals([0, 0, 0], [0, 0, 0]) == [(-np.inf, np.inf)]
    assert workspace.line_intervals([1, 1, 1], [0, 0, 0]) == []
    np.testing.assert_allclose(
        workspace.line_intervals([0, 1, 0], [1, 0, 0]), [(0.0, 0.0)]
    )


def test_base_transform_preserves_line_parameterization() -> None:
    workspace = EllipsoidalWorkspace([1, 0, 0], [0.2, 0.4, 0.6])
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    position = np.array([5, 2, 3])
    placed = workspace.placed(position, rotation)
    assert placed.contains([5, 3, 3])
    np.testing.assert_allclose(
        placed.line_intervals(position, rotation @ np.array([1, 0, 0])),
        [(0.8, 1.2)],
    )
    with pytest.raises(ValueError, match="proper rotation"):
        workspace.placed(position, np.diag([1, 1, -1]))
