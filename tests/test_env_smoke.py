import gymnasium as gym
import mujoco
import numpy as np
from gymnasium.utils.env_checker import check_env

import contact_deflection  # noqa: F401  # Registers ContactDeflection-v0.
from contact_deflection.envs import ContactDeflectionEnv
from contact_deflection.envs.space_robot_env import SpaceRobotEnv
from contact_deflection.estimation.spacetime_line import InterceptCorridor
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


def test_registered_environment_can_be_created() -> None:
    environment = gym.make("ContactDeflection-v0")
    observation, _ = environment.reset(seed=0)
    assert observation.shape == (49,)
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
        for _ in range(10):
            observation, reward, terminated, truncated, _ = environment.step(
                np.full(8, 0.05, dtype=np.float32)
            )
            assert np.all(np.isfinite(observation))
            assert np.isfinite(reward)
            assert not terminated
            assert not truncated
        final_observations.append(observation)
        environment.close()
    np.testing.assert_allclose(final_observations[0], final_observations[1])


def test_scene_uses_vendored_ur5_and_free_base_reacts() -> None:
    environment = ContactDeflectionEnv(workspace=_workspace_at_initial_shield())
    environment.reset(seed=0)
    initial_base_position = environment.spacecraft_position_world
    assert environment.model.nu == 6
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
    for _ in range(5):
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
    for _ in range(5):
        _, _, terminated, _, _ = environment.step(np.zeros(8, dtype=np.float32))
        assert not terminated
    np.testing.assert_allclose(
        environment.projectile_velocity_world, initial_velocity, atol=1e-12
    )
    environment.close()
