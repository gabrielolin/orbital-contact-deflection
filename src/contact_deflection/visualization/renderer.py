"""Deterministic MuJoCo offscreen rendering."""

from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from contact_deflection.envs import ContactDeflectionEnv


def render_smoke_video(
    output_path: str | Path,
    *,
    steps: int = 120,
    seed: int = 0,
    fps: int = 30,
) -> Path:
    """Render a short scripted PD-control rollout to MP4."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = ContactDeflectionEnv(render_mode="rgb_array")
    environment.reset(seed=seed)
    frames: list[np.ndarray] = []
    action = np.array([0.07, -0.12, 0.06, 0.06, -0.03, 0.05])
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
