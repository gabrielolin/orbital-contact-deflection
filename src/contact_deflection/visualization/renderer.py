"""Deterministic MuJoCo offscreen rendering."""

from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from contact_deflection.envs import ContactDeflectionEnv
from contact_deflection.envs.contact_deflection_env import ContactDeflectionConfig
from contact_deflection.envs.space_robot_env import SpaceRobotEnv
from contact_deflection.kinematics.reachable_workspace import EllipsoidalWorkspace


def render_workspace_snapshot(
    output_path: str | Path,
    environment: SpaceRobotEnv,
    workspace: EllipsoidalWorkspace,
) -> Path:
    """Render the smooth contact-candidate envelope."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    placed = workspace.placed(
        environment.spacecraft_position_world,
        environment.spacecraft_rotation_world,
    )
    renderer = mujoco.Renderer(environment.model, height=480, width=640)
    try:
        renderer.update_scene(environment.data, camera="reverse_overview")
        scene = renderer.scene
        if scene.ngeom >= scene.maxgeom:
            raise RuntimeError("MuJoCo visualization scene exhausted geom capacity")
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            placed.radii,
            placed.center,
            placed.rotation.ravel(),
            np.array([0.0, 0.85, 1.0, 0.22], dtype=np.float32),
        )
        scene.ngeom += 1
        imageio.imwrite(output, renderer.render())
    finally:
        renderer.close()
    return output


def render_smoke_video(
    output_path: str | Path,
    task_config: ContactDeflectionConfig,
    *,
    steps: int = 120,
    seed: int = 0,
    fps: int = 30,
) -> Path:
    """Render a short rollout through the canonical structured-action stack."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = ContactDeflectionEnv(task_config, render_mode="rgb_array")
    environment.reset(seed=seed)
    frames: list[np.ndarray] = []
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    stride = max(1, round(1.0 / (fps * environment.config.policy_dt)))
    for step in range(steps):
        _, _, terminated, truncated, _ = environment.step(action)
        if step % stride == 0 or step == steps - 1:
            frame = environment.render()
            if frame is not None:
                frames.append(frame)
        if terminated or truncated:
            break
    environment.close()
    imageio.mimsave(output, frames, fps=fps)  # type: ignore[arg-type]
    return output
