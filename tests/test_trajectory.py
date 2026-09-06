"""Trajectory contract tests independent of any optimization backend."""

import numpy as np
import pytest

from contact_deflection.control.trajectory import (
    JointKinematicState,
    JointTrajectoryLimits,
    PlaceholderTrajectoryGenerator,
)


def _limits() -> JointTrajectoryLimits:
    return JointTrajectoryLimits([-1, -2], [1, 2], [1, 1], [2, 2], [4, 4])


@pytest.mark.parametrize("duration", [0.0, 0.001, 2.0])
def test_placeholder_preserves_impossible_target(duration: float) -> None:
    initial = JointKinematicState([0, 0], [0, 0])
    target = JointKinematicState([50, -50], [100, -100], [0, 0])
    result = PlaceholderTrajectoryGenerator(_limits()).solve(initial, target, duration)
    assert result.trajectory is None
    assert result.requested_duration == duration
    assert result.achieved_duration is None
    assert not result.success
    assert result.status == "not_implemented"
    assert np.all(np.isnan(result.terminal_position_residual))
    assert np.all(np.isnan(result.terminal_velocity_residual))
    np.testing.assert_array_equal(target.q, [50, -50])
    np.testing.assert_array_equal(target.qd, [100, -100])


def test_state_preserves_unknown_acceleration_and_snapshots_input() -> None:
    q = np.zeros(2)
    state = JointKinematicState(q, [0, 0])
    q[:] = 1
    np.testing.assert_array_equal(state.q, [0, 0])
    assert state.qdd is None


@pytest.mark.parametrize("duration", [-1, np.inf, np.nan])
def test_invalid_duration_is_not_a_dynamic_feasibility_failure(duration: float) -> None:
    state = JointKinematicState([0, 0], [0, 0])
    with pytest.raises(ValueError, match="duration"):
        PlaceholderTrajectoryGenerator(_limits()).solve(state, state, duration)


def test_validate_joint_dimensions_and_finite_states() -> None:
    with pytest.raises(ValueError, match="equal shapes"):
        JointKinematicState([0, 0], [0])
    with pytest.raises(ValueError, match="finite"):
        JointKinematicState([np.nan], [0])
    state = JointKinematicState([0], [0])
    with pytest.raises(ValueError, match="joint counts"):
        PlaceholderTrajectoryGenerator(_limits()).solve(state, state, 1)


def test_validate_physical_limit_order_and_sign() -> None:
    with pytest.raises(ValueError, match="strictly below"):
        JointTrajectoryLimits([1], [-1], [1], [1], [1])
    with pytest.raises(ValueError, match="positive"):
        JointTrajectoryLimits([-1], [1], [1], [1], [0])
