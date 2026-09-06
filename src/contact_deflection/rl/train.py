"""Thin Stable-Baselines3 orchestration for the contact benchmark."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import gymnasium as gym

from contact_deflection.envs.contact_deflection_env import (
    ContactDeflectionConfig,
    ContactDeflectionEnv,
)
from contact_deflection.envs.space_robot_env import SpaceRobotEnv
from contact_deflection.kinematics.mink_ik import MinkIK
from contact_deflection.kinematics.reachable_workspace import (
    ReachableWorkspace,
    generate_workspace,
    workspace_fingerprint,
)


@dataclass(frozen=True)
class SACTrainConfig:
    total_timesteps: int = 10_000
    seed: int = 0
    run_dir: str = "outputs/sac"
    evaluation_frequency: int = 5_000
    evaluation_episodes: int = 5


def ensure_workspace(
    task_config: ContactDeflectionConfig,
    cache_path: str | Path,
    *,
    seed: int = 0,
) -> ReachableWorkspace:
    """Load a matching cache or deliberately generate it once for an experiment."""
    path = Path(cache_path)
    robot = SpaceRobotEnv(task_config.robot)
    try:
        robot._reset_physics()
        # Workspace occupancy is position-only; terminal orientation is checked
        # later by the decoder's full-pose IK solve.
        ik = MinkIK(
            robot.model, replace(task_config.decoder.ik, orientation_cost=0.0)
        )
        fingerprint = workspace_fingerprint(
            ik, robot.data.qpos, task_config.decoder.workspace, seed
        )
        if path.exists():
            return ReachableWorkspace.load(path, expected_fingerprint=fingerprint)
        workspace = generate_workspace(
            ik, robot.data.qpos, task_config.decoder.workspace, seed=seed
        )
        if workspace.points.size == 0:
            raise RuntimeError("workspace generation found no feasible shield poses")
        workspace.save(path)
        return workspace
    finally:
        robot.close()


def environment_factory(
    workspace: ReachableWorkspace, task_config: ContactDeflectionConfig
) -> Callable[[], gym.Env]:
    """Return an SB3-compatible factory without sharing simulator state."""
    return lambda: ContactDeflectionEnv(workspace, task_config)


def train_sac(
    workspace: ReachableWorkspace,
    task_config: ContactDeflectionConfig,
    train_config: SACTrainConfig | None = None,
) -> Path:
    """Train SAC on the canonical environment and return the final model path."""
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as error:  # pragma: no cover - exercised by installation docs
        raise ImportError(
            "Install the RL extra with `pip install -e '.[rl]'`."
        ) from error

    settings = train_config or SACTrainConfig()
    run_dir = Path(settings.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    monitor_keys = (
        "contact_success",
        "base_displacement",
        "ik_position_residual",
        "trajectory_feasible",
    )
    environment = DummyVecEnv(
        [
            lambda: Monitor(
                ContactDeflectionEnv(workspace, task_config), info_keywords=monitor_keys
            )
        ]
    )
    evaluation_environment = DummyVecEnv(
        [
            lambda: Monitor(
                ContactDeflectionEnv(workspace, task_config), info_keywords=monitor_keys
            )
        ]
    )
    callback = EvalCallback(
        evaluation_environment,
        best_model_save_path=str(run_dir / "best"),
        log_path=str(run_dir / "evaluations"),
        eval_freq=max(1, settings.evaluation_frequency),
        n_eval_episodes=settings.evaluation_episodes,
        deterministic=True,
    )
    try:
        model = SAC("MlpPolicy", environment, seed=settings.seed, verbose=1)
        model.learn(total_timesteps=settings.total_timesteps, callback=callback)
        model_path = run_dir / "final_model"
        model.save(str(model_path))
        return model_path.with_suffix(".zip")
    finally:
        environment.close()
        evaluation_environment.close()
