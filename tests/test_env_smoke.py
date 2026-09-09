from dataclasses import replace

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium.utils.env_checker import check_env

import contact_deflection  # noqa: F401  # Registers ContactDeflection-v0.
from contact_deflection.envs import ContactDeflectionEnv
from contact_deflection.envs.contact_deflection_env import ContactDeflectionConfig
from contact_deflection.envs.space_robot_env import SpaceRobotEnv
from contact_deflection.estimation.spacetime_line import (
    InterceptCorridor,
    SpacetimeLine,
    intersect_workspace,
)
from contact_deflection.kinematics.reachable_workspace import EllipsoidalWorkspace


def _workspace_at_initial_shield() -> EllipsoidalWorkspace:
    robot = SpaceRobotEnv()
    try:
        robot._reset_physics()
        shield = robot.data.site("shield_center").xpos
        point_B = robot.spacecraft_rotation_world.T @ (
            shield - robot.spacecraft_position_world
        )
        return EllipsoidalWorkspace(point_B, [0.15, 0.15, 0.15])
    finally:
        robot.close()


def test_environment_passes_gymnasium_checker() -> None:
    environment = ContactDeflectionEnv(workspace=_workspace_at_initial_shield())
    check_env(environment, skip_render_check=True)
    environment.close()


def test_benchmark_configuration_loads_canonical_clocks_and_reward() -> None:
    config = ContactDeflectionConfig.load("configs/env.yaml")
    assert config.robot.sim_dt == 0.001
    assert config.robot.control_dt == 0.005
    assert config.robot.policy_dt == 0.250
    assert config.projectile_speed == 3.0
    assert config.desired_velocity_goal.normal_mean == 1.45
    assert config.terminal_reward.follow_through_duration == 0.050
    np.testing.assert_allclose(config.decoder.ik.velocity_limits, [4.0] * 6)
    np.testing.assert_allclose(
        config.decoder.trajectory.acceleration_limits, [15.0] * 6
    )
    np.testing.assert_allclose(config.decoder.trajectory.jerk_limits, [120.0] * 6)
    np.testing.assert_allclose(
        config.controller.torque_limits, [250, 250, 200, 50, 50, 50]
    )


def test_registered_environment_can_be_created() -> None:
    environment = gym.make("ContactDeflection-v0")
    observation, _ = environment.reset(seed=0)
    assert set(observation) == {"observation", "desired_goal"}
    assert observation["observation"].shape == (49,)
    assert observation["desired_goal"].shape == (3,)
    assert environment.action_space.shape == (8,)
    corridor = environment.unwrapped.projectile_intercept_corridor()
    assert isinstance(corridor, InterceptCorridor)
    lower, upper = corridor.intervals[0]
    start = corridor.line.evaluate(lower)[0]
    end = corridor.line.evaluate(upper)[0]
    assert np.linalg.norm(end - start) > 0.45
    environment.close()


def test_reset_and_steps_are_deterministic() -> None:
    environments = [
        ContactDeflectionEnv(workspace=_workspace_at_initial_shield()),
        ContactDeflectionEnv(workspace=_workspace_at_initial_shield()),
    ]
    final_observations = []
    for environment in environments:
        observation, _ = environment.reset(seed=7)
        assert environment.observation_space.contains(observation)
        for _ in range(2):
            observation, reward, terminated, truncated, _ = environment.step(
                np.full(8, 0.05, dtype=np.float32)
            )
            assert all(np.all(np.isfinite(value)) for value in observation.values())
            assert np.isfinite(reward)
            assert not terminated
            assert not truncated
        final_observations.append(observation)
        environment.close()
    for key in final_observations[0]:
        np.testing.assert_allclose(
            final_observations[0][key], final_observations[1][key]
        )


def test_scene_uses_vendored_ur5_and_free_base_reacts() -> None:
    environment = ContactDeflectionEnv(workspace=_workspace_at_initial_shield())
    environment.reset(seed=0)
    initial_base_position = environment.spacecraft_position_world
    assert environment.model.nu == 6
    np.testing.assert_allclose(
        environment.model.actuator_ctrlrange,
        [[-250, 250], [-250, 250], [-200, 200], [-50, 50], [-50, 50], [-50, 50]],
    )
    assert environment.arm_q.shape == (6,)
    assert environment.model.nmesh >= 7
    assert environment.model.body("upper_arm_link").mass[0] == 8.393
    np.testing.assert_allclose(environment.model.geom("bus").size, [0.6, 0.6, 0.6])
    np.testing.assert_allclose(
        environment.model.body("spacecraft_bus").inertia,
        [101.5241013, 101.5241013, 101.5241013],
    )
    assert environment.model.body("spacecraft_bus").mass[0] == 200.0
    np.testing.assert_allclose(
        environment.model.body("manipulator_mount").pos, [0.0, 0.60, 0.0]
    )
    np.testing.assert_allclose(
        environment.arm_q,
        [0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, 0.0],
        atol=0.08,
    )
    shield = environment.data.site("shield_center")
    assert shield.xpos[1] > 0.60
    assert shield.xmat.reshape(3, 3)[1, 2] > 0.99
    assert environment.data.ncon == 0
    solar_panel = environment.model.geom("solar_left_panel")
    np.testing.assert_allclose(solar_panel.size, [1.14, 0.025, 0.35])
    assert solar_panel.group[0] == 2
    assert solar_panel.contype[0] == 0
    assert solar_panel.conaffinity[0] == 0
    assert environment.model.geom("shield_geom").type[0] == mujoco.mjtGeom.mjGEOM_BOX
    np.testing.assert_allclose(
        environment.model.geom("shield_geom").size, [0.20, 0.20, 0.025]
    )
    assert 3.0 < environment.model.body("shield").mass[0] < 6.0
    assert 0.75 < environment.model.body("projectile").mass[0] < 2.25
    assert 0.005 < environment.model.geom("shield_geom").solref[0] < 0.015
    assert 0.05 < environment.model.geom("shield_geom").solref[1] < 0.15
    for _ in range(2):
        environment.step(np.full(8, 0.2, dtype=np.float32))
    assert (
        np.linalg.norm(environment.spacecraft_position_world - initial_base_position)
        > 0.0
    )
    environment.close()


def test_projectile_is_unforced_before_contact() -> None:
    environment = ContactDeflectionEnv(workspace=_workspace_at_initial_shield())
    environment.reset(seed=0)
    initial_velocity = environment.projectile_velocity_world
    for _ in range(2):
        _, _, terminated, _, _ = environment.step(np.zeros(8, dtype=np.float32))
        assert not terminated
    np.testing.assert_allclose(
        environment.projectile_velocity_world, initial_velocity, atol=1e-12
    )
    environment.close()


def test_macro_step_uses_policy_and_control_clocks() -> None:
    environment = ContactDeflectionEnv(workspace=_workspace_at_initial_shield())
    assert environment._physics_steps_per_control == 5
    assert environment._physics_steps_per_action == 250
    environment.reset(seed=0)
    control_calls = 0
    original_compute = environment.controller.compute

    def counting_compute(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal control_calls
        control_calls += 1
        return original_compute(*args, **kwargs)

    environment.controller.compute = counting_compute  # type: ignore[method-assign]
    _, reward, terminated, truncated, _ = environment.step(np.zeros(8))
    assert np.isclose(environment.data.time, 0.250)
    assert control_calls == 50
    assert reward == 0.0
    assert not terminated
    assert not truncated
    environment.close()


def test_goal_sampling_is_seeded_and_varies_across_seeds() -> None:
    environment = ContactDeflectionEnv(workspace=_workspace_at_initial_shield())
    first, first_info = environment.reset(seed=11)
    repeated, repeated_info = environment.reset(seed=11)
    different, different_info = environment.reset(seed=12)
    np.testing.assert_allclose(first["desired_goal"], repeated["desired_goal"])
    assert not np.allclose(first["desired_goal"], different["desired_goal"])
    assert 0.75 <= np.linalg.norm(first["desired_goal"]) <= 2.50
    for key, value in first_info["episode_parameters"].items():
        np.testing.assert_allclose(value, repeated_info["episode_parameters"][key])
    assert first_info["episode_parameters"]["projectile_mass"] != (
        different_info["episode_parameters"]["projectile_mass"]
    )
    assert np.linalg.norm(
        first_info["episode_parameters"]["projectile_velocity_world"]
    ) > 2.5
    np.testing.assert_array_equal(
        first_info["episode_parameters"]["initial_arm_qd"], np.zeros(6)
    )
    assert "projectile_lead_time" not in first_info["episode_parameters"]
    assert not hasattr(environment.task_config, "projectile_lead_time")
    environment.close()


def test_projectile_distribution_crosses_and_spans_workspace() -> None:
    environment = ContactDeflectionEnv()
    normalized_intercepts = []
    try:
        for seed in range(128):
            _, info = environment.reset(seed=seed)
            parameters = info["episode_parameters"]
            line = SpacetimeLine(
                parameters["projectile_initial_position_world"],
                parameters["projectile_velocity_world"],
                0.0,
            )
            workspace = environment._world_workspace()
            corridor = intersect_workspace(
                line,
                workspace,
                horizon=environment.task_config.decoder.prediction_horizon,
            )
            assert isinstance(corridor, InterceptCorridor)
            lower, upper = corridor.intervals[0]
            midpoint = line.evaluate(0.5 * (lower + upper))[0]
            normalized_intercepts.append(
                workspace.rotation.T
                @ (midpoint - workspace.center)
                / workspace.radii
            )
    finally:
        environment.close()

    intercepts = np.asarray(normalized_intercepts)
    assert intercepts[:, 0].min() < -0.7
    assert intercepts[:, 0].max() > 0.7
    assert intercepts[:, 2].min() < -0.7
    assert intercepts[:, 2].max() > 0.7


def test_miss_reward_is_terminal_and_uses_closest_distance() -> None:
    probe = ContactDeflectionEnv(workspace=_workspace_at_initial_shield())
    base = probe.task_config
    probe.close()
    randomization = replace(
        base.episode_randomization,
        projectile_position_offset_mean_shield=(0.0, 0.0, 6.0),
        projectile_position_offset_std_shield=(0.0, 0.0, 0.0),
    )
    config = replace(
        base,
        episode_duration=0.250,
        episode_randomization=randomization,
    )
    environment = ContactDeflectionEnv(
        config=config, workspace=_workspace_at_initial_shield()
    )
    environment.reset(seed=0)
    _, reward, terminated, truncated, info = environment.step(np.zeros(8))
    assert terminated
    assert not truncated
    assert reward < -1.0
    assert info["terminal_reason"] == "miss"
    assert info["miss_distance_penalty"] > 0.0
    environment.close()


def test_contact_runs_terminal_evaluation_and_scores_outgoing_velocity() -> None:
    environment = ContactDeflectionEnv()
    environment.reset(seed=7)
    rewards = []
    for _ in range(8):
        _, reward, terminated, truncated, info = environment.step(np.zeros(8))
        rewards.append(reward)
        if terminated or truncated:
            break
    assert rewards[:-1] == [0.0] * (len(rewards) - 1)
    assert terminated
    assert not truncated
    assert info["terminal_reason"] == "contact"
    assert info["outgoing_velocity_valid"] == 1.0
    assert info["post_contact_separated"] == 1.0
    assert np.linalg.norm(info["outgoing_projectile_velocity_world"]) > 0.25
    assert info["velocity_error_valid"] == 1.0
    assert np.isfinite(reward)
    environment.close()
