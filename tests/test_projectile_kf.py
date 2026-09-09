import numpy as np

from contact_deflection.estimation import ProjectileKalmanFilter


def make_filter() -> ProjectileKalmanFilter:
    return ProjectileKalmanFilter(
        np.array([1.0, 2.0, 3.0, 0.2, -0.1, 0.3]),
        np.eye(6),
        acceleration_noise_std=0.01,
        measurement_noise_std=0.02,
    )


def test_prediction_is_constant_velocity() -> None:
    estimator = make_filter()
    initial_covariance = estimator.covariance.copy()
    dt = 0.5
    transition = np.eye(6)
    transition[:3, 3:] = dt * np.eye(3)
    noise_gain = np.vstack((0.5 * dt**2 * np.eye(3), dt * np.eye(3)))
    expected_covariance = (
        transition @ initial_covariance @ transition.T
        + 0.01**2 * noise_gain @ noise_gain.T
    )
    estimator.predict(dt)
    np.testing.assert_allclose(estimator.mean[:3], [1.1, 1.95, 3.15])
    np.testing.assert_allclose(estimator.mean[3:], [0.2, -0.1, 0.3])
    np.testing.assert_allclose(estimator.covariance, expected_covariance)


def test_prediction_cache_tracks_timestep_and_noise() -> None:
    estimator = make_filter()
    estimator.predict(0.01)
    transition = estimator._transition
    process_covariance = estimator._process_covariance
    estimator.predict(0.01)
    assert estimator._transition is transition
    assert estimator._process_covariance is process_covariance

    estimator.predict(0.02)
    assert estimator._transition is not transition
    assert estimator._process_covariance is not process_covariance


def test_rotating_spacecraft_measurement_update() -> None:
    estimator = make_filter()
    angle = np.pi / 2
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    spacecraft_position = np.array([0.5, -0.5, 0.25])
    measurement = rotation.T @ (estimator.mean[:3] - spacecraft_position)
    estimator.update(measurement, spacecraft_position, rotation)
    np.testing.assert_allclose(estimator.mean, [1, 2, 3, 0.2, -0.1, 0.3])


def test_covariance_remains_symmetric_positive_semidefinite() -> None:
    estimator = make_filter()
    for _ in range(20):
        estimator.predict(0.01)
        estimator.update(np.array([1.0, 2.0, 3.0]), np.zeros(3), np.eye(3))
    np.testing.assert_allclose(estimator.covariance, estimator.covariance.T)
    assert np.linalg.eigvalsh(estimator.covariance).min() >= -1e-12


def test_relative_belief_rotates_mean_and_covariance() -> None:
    estimator = make_filter()
    rotation = np.diag([-1.0, -1.0, 1.0])
    relative = estimator.relative_belief(
        np.array([0.5, 0.5, 0.5]), np.array([0.1, 0.1, 0.1]), rotation
    )
    np.testing.assert_allclose(relative.mean, [-0.5, -1.5, 2.5, -0.1, 0.2, 0.2])
    np.testing.assert_allclose(relative.covariance, np.eye(6))
