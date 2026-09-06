"""Contact intent invariants with independently replaceable IK/trajectory layers."""

import numpy as np
import pytest

from contact_deflection.control.contact_action_decoder import (
    ContactActionDecoder,
    DecodedAction,
    DecoderConfig,
    DecoderState,
)
from contact_deflection.control.trajectory import (
    JointKinematicState,
    TrajectorySolveResult,
)
from contact_deflection.estimation.spacetime_line import (
    InterceptCorridor,
    NoReachableWorkspaceIntersection,
    SpacetimeLine,
)
from contact_deflection.kinematics.mink_ik import IKSolution


class FakeIK:
    velocity_limits = np.full(6, 10.0)

    def __init__(self) -> None:
        self.seeds: list[np.ndarray | None] = []
        self.solution = IKSolution(np.arange(6) * 0.1, 0.2, 0.1, False, {})

    def solve(self, full_qpos, position, orientation, *, seed=None):
        self.seeds.append(None if seed is None else seed.copy())
        return self.solution

    def jacobian(self, full_qpos, q_target):
        return np.eye(6)


class ImpossibleTrajectory:
    def __init__(self) -> None:
        self.calls = []
        self.result = TrajectorySolveResult(
            None,
            0.0,
            None,
            np.full(6, 2.0),
            np.full(6, 3.0),
            False,
            "terminal_constraints_infeasible",
        )

    def solve(self, initial_state, target_state, duration):
        self.calls.append((initial_state, target_state, duration))
        return self.result


def _decoder(config=None):
    ik = FakeIK()
    trajectory = ImpossibleTrajectory()
    return ContactActionDecoder(ik, trajectory, config), ik, trajectory


def _corridor(reference_time=10.0, intervals=((1.0, 3.0),)):
    return InterceptCorridor(
        SpacetimeLine(
            np.array([2.0, 3.0, 4.0]), np.array([-2.0, 0.0, 0.0]), reference_time
        ),
        intervals,
    )


def _state(time=10.0):
    return DecoderState(
        np.zeros(20), JointKinematicState(np.zeros(6), np.zeros(6)), time
    )


@pytest.mark.parametrize("coordinate,tau", [(-1.0, 1.0), (0.0, 2.0), (1.0, 3.0)])
def test_action_selects_one_coupled_spacetime_event(coordinate, tau):
    decoder, _, trajectory = _decoder()
    action = np.zeros(8)
    action[0] = coordinate
    corridor = _corridor()
    decoded = decoder.decode(_state(), corridor, action)
    assert isinstance(decoded, DecodedAction)
    goal = decoded.requested_contact_goal
    assert goal.line_parameter == tau
    assert goal.contact_time == 10.0 + tau
    np.testing.assert_allclose(goal.position_W, corridor.line.evaluate(tau)[0])
    assert trajectory.calls[0][2] == tau


def test_empty_corridor_is_a_normal_result_without_controller_calls():
    decoder, ik, trajectory = _decoder()
    no_intercept = NoReachableWorkspaceIntersection(_corridor().line, "outside")
    assert decoder.decode(_state(), no_intercept, np.zeros(8)) is no_intercept
    assert not ik.seeds
    assert not trajectory.calls


def test_best_effort_failures_and_requested_goal_remain_distinct():
    decoder, ik, trajectory = _decoder()
    decoded = decoder.decode(_state(), _corridor(), np.ones(8))
    assert isinstance(decoded, DecodedAction)
    assert decoded.ik_solution is ik.solution
    assert not decoded.ik_solution.converged
    assert decoded.trajectory_result is trajectory.result
    assert not decoded.trajectory_result.success
    assert decoded.requested_contact_goal.contact_time == 13.0
    np.testing.assert_allclose(decoded.requested_contact_goal.position_W, [-4, 3, 4])
    terminal = trajectory.calls[0][1]
    np.testing.assert_array_equal(terminal.q, ik.solution.q_target)
    np.testing.assert_array_equal(terminal.qd, decoded.requested_joint_velocity)


def test_shield_local_z_and_contact_frame_twist_semantics():
    config = DecoderConfig(
        orientation_scales=(0.2, 0.3),
        linear_velocity_scales=(2, 3, 4),
        linear_velocity_center=(1, -1, 2),
        angular_velocity_scales=(0.5, 0.7),
    )
    decoder, _, _ = _decoder(config)
    action = np.array([0, 0, 0, 0.5, -0.5, 1, 0.4, -1])
    decoded = decoder.decode(_state(), _corridor(), action)
    assert isinstance(decoded, DecodedAction)
    goal, frame = decoded.requested_contact_goal, decoded.contact_frame_W
    np.testing.assert_allclose(frame, np.eye(3))
    np.testing.assert_allclose(goal.shield_normal_W, [1, 0, 0])
    np.testing.assert_allclose(goal.orientation_W[:, 2], frame[:, 0])
    np.testing.assert_allclose(goal.orientation_W.T @ goal.orientation_W, np.eye(3))
    assert np.linalg.det(goal.orientation_W) == pytest.approx(1)
    np.testing.assert_allclose(goal.linear_velocity_W, [2, -2.5, 6])
    np.testing.assert_allclose(goal.angular_velocity_W, [0, 0.2, -0.7])
    requested = np.r_[goal.linear_velocity_W, goal.angular_velocity_W]
    np.testing.assert_allclose(
        decoded.realized_contact_twist + decoded.twist_residual, requested
    )


def test_tilt_uses_world_tangent_axis_and_respects_local_angle():
    decoder, _, _ = _decoder(DecoderConfig(orientation_scales=(0.2, 0.3)))
    action = np.zeros(8)
    action[1] = 1
    goal, _ = decoder.contact_goal(_corridor(), action)
    # +0.2 radians about tangent1 = world +y rotates normal +x towards -z.
    np.testing.assert_allclose(goal.shield_normal_W, [np.cos(0.2), 0, -np.sin(0.2)])


def test_decode_does_not_mutate_action_or_state_and_copies_latent_action():
    decoder, _, _ = _decoder()
    state, corridor, action = _state(), _corridor(), np.zeros(8)
    full_qpos = state.full_qpos.copy()
    decoded = decoder.decode(state, corridor, action)
    assert isinstance(decoded, DecodedAction)
    np.testing.assert_array_equal(state.full_qpos, full_qpos)
    np.testing.assert_array_equal(action, np.zeros(8))
    np.testing.assert_array_equal(state.joints.q, np.zeros(6))
    np.testing.assert_array_equal(corridor.line.position_W0, [2, 3, 4])
    action[:] = 1
    np.testing.assert_array_equal(decoded.latent_action, np.zeros(8))


def test_interval_continuity_uses_absolute_time_and_reset_clears_it():
    decoder, ik, _ = _decoder()
    decoder.decode(_state(), _corridor(intervals=((8.0, 10.0),)), np.zeros(8))
    # Previous event time is 19. New intervals occupy absolute [12,13], [18,20].
    shifted = _corridor(12.0, ((0.0, 1.0), (6.0, 8.0)))
    continued = decoder.decode(_state(12.0), shifted, np.zeros(8))
    assert isinstance(continued, DecodedAction)
    assert continued.requested_contact_goal.contact_time == 19.0
    np.testing.assert_array_equal(ik.seeds[-1], ik.solution.q_target)
    decoder.reset()
    restarted = decoder.decode(_state(12.0), shifted, np.zeros(8))
    assert isinstance(restarted, DecodedAction)
    assert restarted.requested_contact_goal.contact_time == 12.5
    assert ik.seeds[-1] is None


def test_nearby_actions_have_nearby_requested_geometry():
    decoder, _, _ = _decoder(DecoderConfig(angular_velocity_scales=(0.2, 0.3)))
    goal, frame = decoder.contact_goal(_corridor(), np.zeros(8))
    perturbed, perturbed_frame = decoder.contact_goal(_corridor(), np.full(8, 1e-6))
    for first, second in (
        (goal.position_W, perturbed.position_W),
        (goal.orientation_W, perturbed.orientation_W),
        (goal.linear_velocity_W, perturbed.linear_velocity_W),
        (goal.angular_velocity_W, perturbed.angular_velocity_W),
        (frame, perturbed_frame),
    ):
        assert np.linalg.norm(first - second) < 1e-5
    assert abs(goal.contact_time - perturbed.contact_time) < 1e-5


def test_stale_belief_is_explicitly_rejected():
    decoder, _, _ = _decoder()
    with pytest.raises(ValueError, match="propagate belief"):
        decoder.decode(_state(11.0), _corridor(), np.zeros(8))


def test_twist_components_rotate_with_a_non_axis_aligned_contact_frame():
    decoder, _, _ = _decoder(DecoderConfig(angular_velocity_scales=(0.2, 0.3)))
    corridor = InterceptCorridor(
        SpacetimeLine(np.zeros(3), np.array([-1.0, -2.0, -3.0]), 10.0),
        ((1.0, 2.0),),
    )
    action = np.array([0, 0, 0, 1, -1, 0.5, -1, 1])
    goal, frame = decoder.contact_goal(corridor, action)
    np.testing.assert_allclose(frame[:, 0], np.array([1, 2, 3]) / np.sqrt(14))
    np.testing.assert_allclose(frame.T @ goal.linear_velocity_W, [0.5, -0.25, 0.125])
    np.testing.assert_allclose(
        frame.T @ goal.angular_velocity_W, [0, -0.2, 0.3], atol=1e-14
    )


@pytest.mark.parametrize("action", [np.zeros(7), np.full(8, 1.1), np.full(8, np.nan)])
def test_invalid_latents_are_not_silently_clipped(action):
    decoder, _, _ = _decoder()
    with pytest.raises(ValueError, match="Box"):
        decoder.contact_goal(_corridor(), action)
