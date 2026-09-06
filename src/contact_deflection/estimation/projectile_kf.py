"""Linear projectile filter with spacecraft-frame position measurements."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from contact_deflection.estimation.spacetime_line import (
    InterceptCorridor,
    NoReachableWorkspaceIntersection,
    SpacetimeLine,
    WorkspaceIntersection,
    intersect_workspace,
)


@dataclass(frozen=True)
class ProjectileBelief:
    """Gaussian belief over world-frame projectile position and velocity."""

    mean: np.ndarray
    covariance: np.ndarray
    timestamp: float


class ProjectileKalmanFilter:
    """Constant-world-velocity KF with a time-varying body-frame sensor model."""

    def __init__(
        self,
        initial_mean: npt.ArrayLike,
        initial_covariance: npt.ArrayLike,
        *,
        acceleration_noise_std: float,
        measurement_noise_std: float,
        timestamp: float = 0.0,
    ) -> None:
        self.mean = np.asarray(initial_mean, dtype=float).copy()
        self.covariance = np.asarray(initial_covariance, dtype=float).copy()
        if self.mean.shape != (6,) or self.covariance.shape != (6, 6):
            raise ValueError("initial belief must have shapes (6,) and (6, 6)")
        if acceleration_noise_std < 0 or measurement_noise_std <= 0:
            raise ValueError("noise standard deviations must be valid")
        self.acceleration_noise_std = float(acceleration_noise_std)
        self.measurement_covariance = float(measurement_noise_std) ** 2 * np.eye(3)
        self.timestamp = float(timestamp)

    def predict(self, dt: float) -> None:
        """Propagate the world-frame constant-velocity model by ``dt``."""
        if dt <= 0:
            raise ValueError("dt must be positive")
        transition = np.eye(6)
        transition[:3, 3:] = dt * np.eye(3)
        noise_gain = np.vstack((0.5 * dt**2 * np.eye(3), dt * np.eye(3)))
        process_covariance = self.acceleration_noise_std**2 * noise_gain @ noise_gain.T
        self.mean = transition @ self.mean
        self.covariance = (
            transition @ self.covariance @ transition.T + process_covariance
        )
        self.timestamp += dt

    def update(
        self,
        measurement_spacecraft: npt.ArrayLike,
        spacecraft_position_world: npt.ArrayLike,
        rotation_world_from_spacecraft: npt.ArrayLike,
    ) -> None:
        """Update from noisy projectile position expressed in spacecraft axes."""
        measurement = np.asarray(measurement_spacecraft, dtype=float)
        spacecraft_position = np.asarray(spacecraft_position_world, dtype=float)
        rotation = np.asarray(rotation_world_from_spacecraft, dtype=float)
        if measurement.shape != (3,) or spacecraft_position.shape != (3,):
            raise ValueError("positions and measurement must have shape (3,)")
        if rotation.shape != (3, 3):
            raise ValueError("rotation must have shape (3, 3)")

        measurement_matrix = np.zeros((3, 6))
        measurement_matrix[:, :3] = rotation.T
        predicted_measurement = rotation.T @ (self.mean[:3] - spacecraft_position)
        innovation = measurement - predicted_measurement
        innovation_covariance = (
            measurement_matrix @ self.covariance @ measurement_matrix.T
            + self.measurement_covariance
        )
        gain = np.linalg.solve(
            innovation_covariance,
            measurement_matrix @ self.covariance,
        ).T
        self.mean += gain @ innovation
        identity = np.eye(6)
        residual_map = identity - gain @ measurement_matrix
        self.covariance = (
            residual_map @ self.covariance @ residual_map.T
            + gain @ self.measurement_covariance @ gain.T
        )
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    def belief(self) -> ProjectileBelief:
        return ProjectileBelief(
            self.mean.copy(), self.covariance.copy(), self.timestamp
        )

    def spacetime_line(self) -> SpacetimeLine:
        return SpacetimeLine(self.mean[:3], self.mean[3:], self.timestamp)

    def intercept_corridor(
        self, workspace: WorkspaceIntersection, *, horizon: float = 5.0
    ) -> InterceptCorridor | NoReachableWorkspaceIntersection:
        return intersect_workspace(self.spacetime_line(), workspace, horizon=horizon)

    def relative_belief(
        self,
        spacecraft_position_world: npt.ArrayLike,
        spacecraft_velocity_world: npt.ArrayLike,
        rotation_world_from_spacecraft: npt.ArrayLike,
    ) -> ProjectileBelief:
        """Express belief relative to the spacecraft with known spacecraft state."""
        position = np.asarray(spacecraft_position_world, dtype=float)
        velocity = np.asarray(spacecraft_velocity_world, dtype=float)
        rotation = np.asarray(rotation_world_from_spacecraft, dtype=float)
        transform = np.zeros((6, 6))
        transform[:3, :3] = rotation.T
        transform[3:, 3:] = rotation.T
        relative_mean = np.concatenate(
            (
                rotation.T @ (self.mean[:3] - position),
                rotation.T @ (self.mean[3:] - velocity),
            )
        )
        return ProjectileBelief(
            relative_mean,
            transform @ self.covariance @ transform.T,
            self.timestamp,
        )
