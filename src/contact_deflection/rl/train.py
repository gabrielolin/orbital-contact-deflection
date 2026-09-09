"""Thin Stable-Baselines3 orchestration for the contact benchmark."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym

from contact_deflection.envs.contact_deflection_env import (
    ContactDeflectionConfig,
    ContactDeflectionEnv,
)


@dataclass(frozen=True)
class SACTrainConfig:
    total_timesteps: int = 10_000
    seed: int = 0
    run_dir: str = "outputs/sac"
    evaluation_frequency: int = 5_000
    evaluation_episodes: int = 5
    # The fixed 0.01 probe under-explored. The successful automatic-entropy
    # probe converged near 0.325, so initialize adaptive tuning close to 0.3.
    entropy_coefficient: str | float = "auto_0.3"
    progress_bar: bool = True
    wandb_enabled: bool = False


def environment_factory(
    task_config: ContactDeflectionConfig,
) -> Callable[[], gym.Env]:
    """Return an SB3-compatible factory without sharing simulator state."""
    return lambda: ContactDeflectionEnv(task_config)


def train_sac(
    task_config: ContactDeflectionConfig,
    train_config: SACTrainConfig | None = None,
) -> Path:
    """Train SAC on the canonical environment and return the final model path."""
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
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
        "ik_converged",
        "ik_position_residual",
        "trajectory_feasible",
        "velocity_error",
        "velocity_score",
        "miss_distance_penalty",
        "spacecraft_angular_momentum_delta",
        "angular_momentum_penalty",
        "post_contact_separated",
    )
    environment = DummyVecEnv(
        [lambda: Monitor(ContactDeflectionEnv(task_config), info_keywords=monitor_keys)]
    )
    evaluation_environment = DummyVecEnv(
        [lambda: Monitor(ContactDeflectionEnv(task_config), info_keywords=monitor_keys)]
    )
    evaluation_callback = EvalCallback(
        evaluation_environment,
        best_model_save_path=str(run_dir / "best"),
        log_path=str(run_dir / "evaluations"),
        eval_freq=max(1, settings.evaluation_frequency),
        n_eval_episodes=settings.evaluation_episodes,
        deterministic=True,
    )
    callbacks: list[BaseCallback] = [evaluation_callback]
    tensorboard_log: str | None = None
    if settings.wandb_enabled:
        try:
            import wandb
            from wandb.integration.sb3 import WandbCallback
        except ImportError as error:  # pragma: no cover - installation error path
            raise ImportError(
                "Install tracking support with `pip install -e '.[tracking]'`."
            ) from error
        if wandb.run is None:
            raise RuntimeError("wandb_enabled requires an active wandb.init() run")
        callbacks.append(WandbCallback(gradient_save_freq=0, model_save_freq=0))
        tensorboard_log = str(run_dir / "tensorboard")
    try:
        model = SAC(
            "MultiInputPolicy",
            environment,
            seed=settings.seed,
            ent_coef=settings.entropy_coefficient,
            tensorboard_log=tensorboard_log,
            verbose=1,
        )
        model.learn(
            total_timesteps=settings.total_timesteps,
            callback=callbacks,
            progress_bar=settings.progress_bar,
        )
        model_path = run_dir / "final_model"
        model.save(str(model_path))
        return model_path.with_suffix(".zip")
    finally:
        environment.close()
        evaluation_environment.close()
