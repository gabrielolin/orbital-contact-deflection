"""Smooth Cartesian candidate region for online contact-pose IK.

The ellipsoid is an action-parameterization envelope, not a reachability
certificate. It is configured conservatively in spacecraft coordinates and
placed at the current floating-base pose. Every selected contact pose is still
checked by the decoder's online constrained IK solve.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


def _vector(value: npt.ArrayLike, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite three-vector")
    return result


def _rotation(value: npt.ArrayLike) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if (
        result.shape != (3, 3)
        or not np.all(np.isfinite(result))
        or not np.allclose(result.T @ result, np.eye(3), atol=1e-8)
        or not np.isclose(np.linalg.det(result), 1.0, atol=1e-8)
    ):
        raise ValueError("rotation must be a proper rotation matrix")
    return result


@dataclass(frozen=True)
class WorkspaceConfig:
    """Axis-aligned ellipsoid in the spacecraft frame.

    For the canonical front-mounted arm, ``radii[1]`` is the direct knob for
    the front/back extent and therefore for a near-normal projectile corridor.
    """

    center: tuple[float, float, float] = (0.30, 1.30, 0.0)
    radii: tuple[float, float, float] = (0.50, 0.275, 0.50)

    def __post_init__(self) -> None:
        _vector(self.center, "workspace center")
        radii = _vector(self.radii, "workspace radii")
        if np.any(radii <= 0):
            raise ValueError("workspace radii must be positive")

    def build(self) -> "EllipsoidalWorkspace":
        return EllipsoidalWorkspace(self.center, self.radii)


class EllipsoidalWorkspace:
    """Smooth ellipsoidal candidate region with analytic line intersection."""

    def __init__(
        self,
        center: npt.ArrayLike,
        radii: npt.ArrayLike,
        rotation: npt.ArrayLike | None = None,
    ) -> None:
        self.center = _vector(center, "center").copy()
        self.radii = _vector(radii, "radii").copy()
        if np.any(self.radii <= 0):
            raise ValueError("radii must be positive")
        self.rotation = _rotation(np.eye(3) if rotation is None else rotation).copy()
        for value in (self.center, self.radii, self.rotation):
            value.setflags(write=False)

    def _local(self, position: npt.ArrayLike) -> np.ndarray:
        return self.rotation.T @ (_vector(position, "position") - self.center)

    def contains(self, position: npt.ArrayLike) -> bool:
        local = self._local(position)
        return bool(np.dot(local / self.radii, local / self.radii) <= 1.0 + 1e-12)

    def line_intervals(
        self, origin: npt.ArrayLike, direction: npt.ArrayLike
    ) -> list[tuple[float, float]]:
        """Return the line parameter interval inside the ellipsoid.

        Direction need not be normalized. Supplying velocity therefore gives
        an interval in seconds, matching ``SpacetimeLine``.
        """
        local_origin = self._local(origin) / self.radii
        local_direction = self.rotation.T @ _vector(direction, "direction")
        local_direction = local_direction / self.radii
        quadratic = float(local_direction @ local_direction)
        constant = float(local_origin @ local_origin - 1.0)
        if quadratic <= 1e-24:
            return [(-np.inf, np.inf)] if constant <= 0 else []
        linear = 2.0 * float(local_origin @ local_direction)
        discriminant = linear**2 - 4.0 * quadratic * constant
        if discriminant < -1e-12:
            return []
        root = np.sqrt(max(0.0, discriminant))
        lower = (-linear - root) / (2.0 * quadratic)
        upper = (-linear + root) / (2.0 * quadratic)
        return [(lower, upper)]

    def placed(
        self, position_W: npt.ArrayLike, rotation_WB: npt.ArrayLike
    ) -> "EllipsoidalWorkspace":
        base_position = _vector(position_W, "base position")
        base_rotation = _rotation(rotation_WB)
        return EllipsoidalWorkspace(
            base_position + base_rotation @ self.center,
            self.radii,
            base_rotation @ self.rotation,
        )
