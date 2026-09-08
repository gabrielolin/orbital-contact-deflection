import numpy as np

from contact_deflection.estimation.projectile_kf import ProjectileKalmanFilter
from contact_deflection.estimation.spacetime_line import (
    InterceptCorridor,
    NoReachableWorkspaceIntersection,
    SpacetimeLine,
    intersect_workspace,
)
from contact_deflection.geometry.contact_frame import contact_frame
from contact_deflection.kinematics.reachable_workspace import EllipsoidalWorkspace


def test_kf_line_and_intersection_keep_seconds_not_distance():
    kf = ProjectileKalmanFilter(
        [0, 0, 0, 2, 0, 0],
        np.eye(6),
        acceleration_noise_std=0,
        measurement_noise_std=0.01,
        timestamp=10,
    )
    kf.predict(0.5)
    line = kf.spacetime_line()
    p, t = line.evaluate(2)
    np.testing.assert_allclose(p, [5, 0, 0])
    assert t == 12.5
    np.testing.assert_allclose(line.tangent_xyz(), [1, 0, 0])
    workspace = EllipsoidalWorkspace([3.0, 0, 0], [0.5, 0.5, 0.5])
    corridor = kf.intercept_corridor(workspace)
    assert isinstance(corridor, InterceptCorridor)
    np.testing.assert_allclose(corridor.intervals, [[0.75, 1.25]])


def test_stationary_future_and_empty_corridors():
    workspace = EllipsoidalWorkspace([0, 0, 0], [0.1, 0.1, 0.1])
    stationary = SpacetimeLine(np.zeros(3), np.zeros(3), 0)
    corridor = intersect_workspace(stationary, workspace, horizon=3)
    assert isinstance(corridor, InterceptCorridor)
    assert corridor.intervals == ((0.0, 3.0),)
    away = SpacetimeLine(np.ones(3), np.ones(3), 0)
    assert isinstance(
        intersect_workspace(away, workspace), NoReachableWorkspaceIntersection
    )


def test_contact_frame_transport_crosses_axis_tie_without_jump():
    frame = contact_frame(np.array([1.0, 1.00001, 1.0]))
    next_frame = contact_frame(np.array([1.00001, 1.0, 1.0]), frame)
    np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-12)
    assert np.linalg.det(frame) > 0.999999
    np.testing.assert_allclose(
        frame[:, 0],
        -np.array([1.0, 1.00001, 1.0]) / np.linalg.norm([1.0, 1.00001, 1.0]),
    )
    assert np.linalg.norm(frame - next_frame) < 1e-4
    np.testing.assert_array_equal(contact_frame(np.zeros(3), frame), frame)
