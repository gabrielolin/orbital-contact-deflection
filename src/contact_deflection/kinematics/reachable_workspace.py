"""Cached sampled Cartesian workspace; no dynamic/time reachability claims.

Accepted constrained-IK samples are stored in spacecraft coordinates. A small
union of balls supplies a usable occupancy approximation between samples. Only
the sample centers are validated: ball neighborhoods are NOT certified feasible.
Terminal IK must still check each requested pose (including its orientation).
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial import cKDTree

from contact_deflection.kinematics.mink_ik import MinkIK


def _vector(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError("Expected a finite three-vector")
    return result


@dataclass(frozen=True)
class WorkspaceConfig:
    sample_count: int = 1000
    bounds_lower: tuple[float, float, float] = (-1.0, -1.0, 0.3)
    bounds_upper: tuple[float, float, float] = (1.0, 1.0, 2.0)
    radius: float = 0.025
    cache_path: str = "outputs/reachable_workspace.npz"

    def __post_init__(self) -> None:
        if self.sample_count < 1 or int(self.sample_count) != self.sample_count:
            raise ValueError("sample_count must be a positive integer")
        if np.any(_vector(np.asarray(self.bounds_lower)) >= self.bounds_upper):
            raise ValueError("Sampling lower bounds must be below upper bounds")
        _vector(np.asarray(self.bounds_upper))
        if not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("Workspace radius must be finite and positive")


class ReachableWorkspace:
    """Union of sampled balls with exact line intersection in its stored frame.

    Use ``placed(base_position_W, rotation_WB)`` on a base-frame cache before
    intersecting a world-frame projectile line. This freezes the current base
    pose for approximate geometry; it does not predict arm-induced base motion.
    """

    def __init__(
        self,
        points_B: np.ndarray,
        radius: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        points = np.asarray(points_B, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Workspace points must have shape (N, 3)")
        if not np.all(np.isfinite(points)):
            raise ValueError("Workspace points must be finite")
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("Workspace radius must be finite and positive")
        self.points = points.copy()
        self.points.setflags(write=False)
        self.radius = float(radius)
        self.metadata = dict(metadata or {})
        self._tree = cKDTree(self.points)

    def contains(self, position_W: np.ndarray) -> bool:
        distance, _ = self._tree.query(_vector(position_W))
        return bool(distance <= self.radius)

    def line_intervals(
        self, origin_W: np.ndarray, direction_W: np.ndarray
    ) -> list[tuple[float, float]]:
        """Parameters s with origin + s*direction inside, including tangencies.

        Direction need not be normalized (velocity yields intervals in seconds).
        A stationary in-workspace line occupies all real parameters.
        """
        origin, direction = _vector(origin_W), _vector(direction_W)
        speed = float(np.linalg.norm(direction))
        if speed == 0:
            return [(-np.inf, np.inf)] if self.contains(origin) else []
        axis = direction / speed
        offsets = self.points - origin
        projection = offsets @ axis
        perpendicular = offsets - projection[:, None] * axis
        discriminant = self.radius**2 - np.sum(perpendicular**2, axis=1)
        valid = discriminant >= 0
        half_width = np.sqrt(discriminant[valid]) / speed
        center = projection[valid] / speed
        intervals = sorted(zip(center - half_width, center + half_width, strict=True))
        merged: list[tuple[float, float]] = []
        for lower, upper in intervals:
            if merged and lower <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], float(upper)))
            else:
                merged.append((float(lower), float(upper)))
        return merged

    def placed(
        self, position_W: np.ndarray, rotation_WB: np.ndarray
    ) -> "ReachableWorkspace":
        rotation = np.asarray(rotation_WB, dtype=float)
        if (
            rotation.shape != (3, 3)
            or not np.all(np.isfinite(rotation))
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8)
        ):
            raise ValueError("rotation_WB must be a proper rotation")
        return ReachableWorkspace(
            self.points @ rotation.T + _vector(position_W),
            self.radius,
            {**self.metadata, "coordinate_frame": "world"},
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as stream:
            np.savez_compressed(
                stream,
                points=self.points,
                radius=self.radius,
                metadata=json.dumps(self.metadata, sort_keys=True, allow_nan=False),
                version=1,
            )

    @classmethod
    def load(
        cls, path: str | Path, expected_fingerprint: str | None = None
    ) -> "ReachableWorkspace":
        with np.load(path, allow_pickle=False) as data:
            if int(data["version"]) != 1:
                raise ValueError("Unsupported workspace cache version")
            metadata = json.loads(str(data["metadata"]))
            if (
                expected_fingerprint is not None
                and metadata.get("fingerprint") != expected_fingerprint
            ):
                raise ValueError("Workspace cache model/configuration mismatch")
            return cls(data["points"], float(data["radius"]), metadata)

    def export_xyz(self, path: str | Path) -> None:
        """Export accepted sample centers for plotting without another IK job."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(target, self.points, delimiter=",", header="x,y,z", comments="")


def workspace_fingerprint(
    solver: MinkIK, full_qpos: np.ndarray, config: WorkspaceConfig, seed: int
) -> str:
    """Hash compiled model, sampling/IK settings, starting state and seed."""
    buffer = np.empty(mujoco.mj_sizeModel(solver.model), dtype=np.uint8)
    mujoco.mj_saveModel(solver.model, buffer=buffer)
    digest = hashlib.sha256(buffer.tobytes())
    digest.update(np.asarray(full_qpos, dtype=np.float64).tobytes())
    settings = {"workspace": asdict(config), "ik": asdict(solver.config), "seed": seed}
    digest.update(json.dumps(settings, sort_keys=True).encode())
    return digest.hexdigest()


def generate_workspace(
    solver: MinkIK,
    full_qpos: np.ndarray,
    config: WorkspaceConfig | None = None,
    *,
    seed: int = 0,
) -> ReachableWorkspace:
    """Test nominal plus uniform Cartesian candidates with constrained Mink IK.

    Solver tolerances and collision clearance come from its IKConfig. For a
    position-only workspace pass an IKConfig with orientation_cost=0. A failed
    candidate never replaces the latest accepted warm start. No environment
    state is changed. Saving is explicit so generation can be inspected first.
    """
    settings = config or WorkspaceConfig()
    qpos = np.asarray(full_qpos, dtype=float).copy()
    if qpos.shape != (solver.model.nq,) or not np.all(np.isfinite(qpos)):
        raise ValueError("full_qpos must match the model and be finite")
    data = mujoco.MjData(solver.model)
    data.qpos[:] = qpos
    mujoco.mj_forward(solver.model, data)
    bus = solver.model.body("spacecraft_bus").id
    base_position = data.xpos[bus].copy()
    rotation = data.xmat[bus].reshape(3, 3).copy()
    nominal_position, nominal_orientation = solver.pose(qpos)
    candidates = np.random.default_rng(seed).uniform(
        settings.bounds_lower, settings.bounds_upper, (settings.sample_count, 3)
    )
    candidates = np.vstack(((nominal_position - base_position) @ rotation, candidates))
    accepted = []
    residuals = []
    warm_start = qpos[solver.arm_qpos_indices].copy()
    for candidate in candidates:
        requested = base_position + rotation @ candidate
        solution = solver.solve(qpos, requested, nominal_orientation, seed=warm_start)
        residuals.append(float(solution.position_residual))
        if solution.converged and solution.metadata.get("constraints_satisfied", False):
            achieved, _ = solver.pose(qpos, solution.q_target)
            accepted.append((achieved - base_position) @ rotation)
            warm_start = solution.q_target.copy()
    finite = np.asarray(residuals)[np.isfinite(residuals)]
    return ReachableWorkspace(
        np.asarray(accepted).reshape(-1, 3),
        settings.radius,
        {
            "coordinate_frame": "spacecraft_bus",
            "approximation": "union of balls; only centers validated by IK",
            "fingerprint": workspace_fingerprint(solver, qpos, settings, seed),
            "workspace_config": asdict(settings),
            "ik_config": asdict(solver.config),
            "seed": seed,
            "candidate_count": len(candidates),
            "accepted_count": len(accepted),
            "acceptance_rate": len(accepted) / len(candidates),
            "position_residual_mean": float(finite.mean()) if finite.size else None,
            "position_residual_max": float(finite.max()) if finite.size else None,
            "nonfinite_residual_count": len(residuals) - len(finite),
        },
    )
