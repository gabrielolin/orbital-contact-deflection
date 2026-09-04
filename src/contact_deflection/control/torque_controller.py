"""Basic joint-space PD torque controller."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class PDControllerConfig:
    kp: npt.ArrayLike
    kd: npt.ArrayLike
    torque_limits: npt.ArrayLike


class TorqueController:
    """Version-one PD controller with symmetric actuator torque clipping."""

    def __init__(self, config: PDControllerConfig) -> None:
        self.kp = np.asarray(config.kp, dtype=float)
        self.kd = np.asarray(config.kd, dtype=float)
        self.torque_limits = np.asarray(config.torque_limits, dtype=float)
        if not (self.kp.shape == self.kd.shape == self.torque_limits.shape):
            raise ValueError("controller parameter arrays must have equal shapes")
        if np.any(self.kp < 0) or np.any(self.kd < 0):
            raise ValueError("PD gains must be nonnegative")
        if np.any(self.torque_limits <= 0):
            raise ValueError("torque limits must be positive")

    def compute(
        self,
        q: npt.ArrayLike,
        qd: npt.ArrayLike,
        q_ref: npt.ArrayLike,
        qd_ref: npt.ArrayLike,
        qdd_ref: npt.ArrayLike | None = None,
    ) -> np.ndarray:
        """Compute clipped feedback torque; qdd_ref is reserved for feedforward."""
        del qdd_ref
        q_array = np.asarray(q, dtype=float)
        qd_array = np.asarray(qd, dtype=float)
        q_ref_array = np.asarray(q_ref, dtype=float)
        qd_ref_array = np.asarray(qd_ref, dtype=float)
        expected = self.kp.shape
        if any(
            x.shape != expected for x in (q_array, qd_array, q_ref_array, qd_ref_array)
        ):
            raise ValueError(f"joint arrays must have shape {expected}")
        torque = self.kp * (q_ref_array - q_array) + self.kd * (qd_ref_array - qd_array)
        return np.clip(torque, -self.torque_limits, self.torque_limits)
