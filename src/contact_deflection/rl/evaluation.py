"""Deterministic SAC evaluation and qualitative rollout rendering."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import mujoco
import numpy as np
import numpy.typing as npt

from contact_deflection.envs.contact_deflection_env import (
    ContactDeflectionConfig,
    ContactDeflectionEnv,
)
from contact_deflection.estimation.spacetime_line import InterceptCorridor


class PredictPolicy(Protocol):
    """Minimal SB3-compatible deterministic inference interface."""

    def predict(
        self, observation: dict[str, np.ndarray], *, deterministic: bool
    ) -> tuple[np.ndarray, Any]: ...


def _episode_metrics(
    seed: int,
    episode_return: float,
    info: dict[str, Any],
) -> dict[str, Any]:
    velocity_valid = bool(info["velocity_error_valid"])
    return {
        "seed": seed,
        "return": episode_return,
        "contact": bool(info["contact_success"]),
        "velocity_error": float(info["velocity_error"]) if velocity_valid else None,
        "velocity_score": float(info["velocity_score"]),
        "base_displacement": float(info["base_displacement"]),
        "angular_momentum_delta": float(
            info["spacecraft_angular_momentum_delta"]
        ),
        "ik_converged": bool(info["ik_converged"]),
        "trajectory_feasible": bool(info["trajectory_feasible"]),
        "post_contact_separated": bool(info["post_contact_separated"]),
        "terminal_reason": str(info["terminal_reason"]),
        "desired_velocity_world": np.asarray(
            info["desired_projectile_velocity_world"]
        ).tolist(),
        "outgoing_velocity_world": np.asarray(
            info["outgoing_projectile_velocity_world"]
        ).tolist(),
        "episode_parameters": {
            key: np.asarray(value).tolist()
            for key, value in info["episode_parameters"].items()
        },
    }


def _aggregate(
    episodes: Sequence[dict[str, Any]],
) -> dict[str, float | int | None]:
    returns = np.asarray([episode["return"] for episode in episodes], dtype=float)
    velocity_errors = [
        episode["velocity_error"]
        for episode in episodes
        if episode["velocity_error"] is not None
    ]
    return {
        "episodes": len(episodes),
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "contact_rate": float(np.mean([episode["contact"] for episode in episodes])),
        "velocity_error_mean": (
            float(np.mean(velocity_errors)) if velocity_errors else None
        ),
        "base_displacement_mean": float(
            np.mean([episode["base_displacement"] for episode in episodes])
        ),
        "angular_momentum_delta_mean": float(
            np.mean([episode["angular_momentum_delta"] for episode in episodes])
        ),
        "ik_convergence_rate": float(
            np.mean([episode["ik_converged"] for episode in episodes])
        ),
        "trajectory_feasibility_rate": float(
            np.mean([episode["trajectory_feasible"] for episode in episodes])
        ),
    }


def evaluate_policy(
    policy: PredictPolicy,
    task_config: ContactDeflectionConfig,
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Evaluate deterministic policy actions on an explicit seed set."""
    if not seeds:
        raise ValueError("evaluation requires at least one seed")
    environment = ContactDeflectionEnv(task_config)
    episodes: list[dict[str, Any]] = []
    try:
        for seed in seeds:
            observation, _ = environment.reset(seed=int(seed))
            episode_return = 0.0
            while True:
                action, _ = policy.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = environment.step(
                    action
                )
                episode_return += float(reward)
                if terminated or truncated:
                    break
            episodes.append(_episode_metrics(int(seed), episode_return, info))
    finally:
        environment.close()
    return {"aggregate": _aggregate(episodes), "episodes": episodes}


class RecordingContactEnv(ContactDeflectionEnv):
    """Record physics-rate rollout state at video rate with intent overlays."""

    def __init__(self, config: ContactDeflectionConfig, fps: int = 30) -> None:
        super().__init__(config, render_mode="rgb_array")
        self.frames: list[np.ndarray] = []
        self.fps = fps
        self._capture_stride = max(1, round(1.0 / (fps * self.config.sim_dt)))
        self._physics_count = 0
        self._recording = False
        self._goal_arrow_origin_world = np.zeros(3)

    def start_recording(self) -> None:
        self.frames.clear()
        self._physics_count = 0
        self._recording = True
        self._goal_arrow_origin_world = (
            self.data.site("shield_center").xpos.copy() + np.array([0.0, 0.0, 0.30])
        )
        self.frames.append(self._render_debug_frame())

    @staticmethod
    def _add_connector(
        scene: mujoco.MjvScene,
        geom_type: mujoco.mjtGeom,
        width: float,
        start: np.ndarray,
        end: np.ndarray,
        color: list[float],
    ) -> None:
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            geom_type,
            np.ones(3),
            np.zeros(3),
            np.eye(3).ravel(),
            np.asarray(color, dtype=np.float32),
        )
        mujoco.mjv_connector(geom, geom_type, width, start, end)
        scene.ngeom += 1

    @staticmethod
    def _add_sphere(
        scene: mujoco.MjvScene,
        position: np.ndarray,
        radius: float,
        color: list[float],
    ) -> None:
        if scene.ngeom >= scene.maxgeom:
            return
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.full(3, radius),
            position,
            np.eye(3).ravel(),
            np.asarray(color, dtype=np.float32),
        )
        scene.ngeom += 1

    def _render_debug_frame(self) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera="reverse_overview")
        scene = self._renderer.scene
        corridor_display_offset = np.zeros(3)
        if isinstance(self._last_corridor, InterceptCorridor):
            lower, upper = self._last_corridor.intervals[0]
            line_start = self._last_corridor.line.evaluate(lower)[0]
            line_end = self._last_corridor.line.evaluate(upper)[0]
            midpoint = 0.5 * (line_start + line_end)
            camera_position = self.model.camera("reverse_overview").pos
            camera_direction = camera_position - midpoint
            corridor_display_offset = 0.06 * camera_direction / np.linalg.norm(
                camera_direction
            )
            display_start = line_start + corridor_display_offset
            display_end = line_end + corridor_display_offset
            self._add_connector(
                scene,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                0.012,
                display_start,
                display_end,
                [0.2, 1.0, 0.25, 0.85],
            )
            self._add_sphere(scene, display_start, 0.022, [0.2, 1.0, 0.25, 0.9])
            self._add_sphere(scene, display_end, 0.022, [0.2, 1.0, 0.25, 0.9])
        self._add_connector(
            scene,
            mujoco.mjtGeom.mjGEOM_ARROW,
            0.025,
            self._goal_arrow_origin_world,
            self._goal_arrow_origin_world + 0.5 * self._desired_velocity_world,
            [0.0, 0.9, 1.0, 0.95],
        )
        if self._last_decoded is not None:
            self._add_sphere(
                scene,
                self._last_decoded.requested_contact_goal.position_W
                + corridor_display_offset,
                0.050,
                [1.0, 0.85, 0.0, 0.98],
            )
        return self._renderer.render().copy()

    def _step_physics(self, torque: npt.ArrayLike) -> None:
        super()._step_physics(torque)
        self._physics_count += 1
        if self._recording and self._physics_count % self._capture_stride == 0:
            self.frames.append(self._render_debug_frame())

    def record_aftermath(self, seconds: float) -> None:
        """Hold the arm at its terminal pose while recording the aftermath."""
        if not np.isfinite(seconds) or seconds < 0:
            raise ValueError("post-contact render duration must be nonnegative")
        hold_q = self.arm_q
        torque = np.zeros(self.model.nu)
        for step_index in range(round(seconds / self.config.sim_dt)):
            if step_index % self._physics_steps_per_control == 0:
                torque = self.controller.compute(
                    self.arm_q, self.arm_qd, hold_q, np.zeros(6)
                )
            self._step_physics(torque)


def render_policy_episode(
    policy: PredictPolicy,
    task_config: ContactDeflectionConfig,
    output_path: str | Path,
    *,
    seed: int,
    post_contact_seconds: float = 4.0,
    fps: int = 30,
) -> dict[str, Any]:
    """Render one deterministic episode and return its metrics."""
    try:
        import imageio.v2 as imageio
    except ImportError as error:  # pragma: no cover - installation error path
        raise ImportError(
            "Install the RL extra with `pip install -e '.[rl]'`."
        ) from error
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = RecordingContactEnv(task_config, fps=fps)
    try:
        observation, _ = environment.reset(seed=seed)
        environment.start_recording()
        episode_return = 0.0
        while True:
            action, _ = policy.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = environment.step(action)
            episode_return += float(reward)
            if terminated or truncated:
                break
        if info["contact_success"]:
            environment.record_aftermath(post_contact_seconds)
        imageio.mimsave(output, environment.frames, fps=fps)  # type: ignore[arg-type]
        result = _episode_metrics(seed, episode_return, info)
        result["video"] = str(output)
        result["rendered_post_contact_seconds"] = (
            post_contact_seconds if info["contact_success"] else 0.0
        )
        return result
    finally:
        environment.close()


def latest_checkpoint(run_dir: str | Path) -> Path:
    """Return the most recently modified SB3 zip checkpoint under a run."""
    candidates = list(Path(run_dir).rglob("*.zip"))
    if not candidates:
        raise FileNotFoundError(f"no .zip checkpoints found under {run_dir}")
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))
