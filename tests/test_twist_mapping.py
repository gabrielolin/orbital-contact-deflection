import numpy as np

from contact_deflection.control.twist_mapping import map_terminal_twist


def test_feasible_twist_is_reproduced() -> None:
    requested = np.arange(6) * 0.05
    result = map_terminal_twist(np.eye(6), requested, np.ones(6))
    assert result.success
    np.testing.assert_allclose(result.realized_twist, requested, atol=1e-6)
    assert not np.any(result.saturated)


def test_saturation_and_singular_jacobian_are_explicit() -> None:
    result = map_terminal_twist(np.eye(6), np.ones(6) * 3, np.ones(6))
    assert result.success
    assert np.all(result.saturated)
    assert np.linalg.norm(result.twist_residual) > 1
    np.testing.assert_allclose(
        result.requested_twist, result.realized_twist + result.twist_residual
    )
    singular = map_terminal_twist(np.zeros((6, 6)), np.ones(6), np.ones(6))
    assert np.all(np.isfinite(singular.qdot_target))
    np.testing.assert_allclose(singular.realized_twist, 0)
    np.testing.assert_allclose(singular.twist_residual, 1)
