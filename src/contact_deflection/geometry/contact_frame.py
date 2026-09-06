"""Contact frame columns are [incoming-opposing normal, tangent1, tangent2]."""

import numpy as np


def contact_frame(
    velocity_W: np.ndarray, previous: np.ndarray | None = None
) -> np.ndarray:
    velocity = np.asarray(velocity_W, dtype=float)
    if velocity.shape != (3,) or not np.all(np.isfinite(velocity)):
        raise ValueError("velocity must be a finite 3-vector")
    speed = np.linalg.norm(velocity)
    if speed <= 1e-12:
        return np.eye(3) if previous is None else previous.copy()
    normal = -velocity / speed
    # Transport the previous tangent rather than switching basis at ties.
    tangent = np.zeros(3)
    if previous is not None:
        tangent = previous[:, 1] - normal * (normal @ previous[:, 1])
    if np.linalg.norm(tangent) <= 1e-8:
        axis = np.eye(3)[np.argmin(np.abs(normal))]
        tangent = axis - normal * (axis @ normal)
    tangent /= np.linalg.norm(tangent)
    return np.column_stack((normal, tangent, np.cross(normal, tangent)))
