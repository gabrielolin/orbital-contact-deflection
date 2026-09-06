"""Bounded terminal twist fitting with explicit realization diagnostics."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear


@dataclass(frozen=True)
class TwistMappingResult:
    qdot_target: np.ndarray
    requested_twist: np.ndarray
    realized_twist: np.ndarray
    twist_residual: np.ndarray
    saturated: np.ndarray
    success: bool
    status: str


def map_terminal_twist(
    jacobian: np.ndarray,
    requested_twist: np.ndarray,
    velocity_limits: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    regularization: float = 1e-6,
) -> TwistMappingResult:
    """Fit [linear, angular] world twist; weights are diagonal objective weights.

    Jacobian semantics are the caller's responsibility. The initial Mink adapter
    supplies a frozen-base arm Jacobian, not a momentum-coupled prediction.
    """
    jacobian = np.asarray(jacobian, dtype=float)
    requested = np.asarray(requested_twist, dtype=float)
    limits = np.asarray(velocity_limits, dtype=float)
    weight = np.ones(6) if weights is None else np.asarray(weights, dtype=float)
    if (
        jacobian.ndim != 2
        or jacobian.shape[0] != 6
        or requested.shape != (6,)
        or limits.shape != (jacobian.shape[1],)
        or weight.shape != (6,)
        or not all(
            np.all(np.isfinite(x)) for x in (jacobian, requested, limits, weight)
        )
        or np.any(limits <= 0)
        or np.any(weight < 0)
        or not np.isfinite(regularization)
        or regularization <= 0
    ):
        raise ValueError("Invalid twist problem dimensions, weights, or bounds")
    root_weight = np.sqrt(weight)
    matrix = np.vstack(
        (root_weight[:, None] * jacobian, np.sqrt(regularization) * np.eye(limits.size))
    )
    rhs = np.concatenate((root_weight * requested, np.zeros(limits.size)))
    fit = lsq_linear(matrix, rhs, bounds=(-limits, limits), tol=1e-10)
    qdot = np.asarray(fit.x)
    realized = jacobian @ qdot
    return TwistMappingResult(
        qdot,
        requested.copy(),
        realized,
        requested - realized,
        np.isclose(np.abs(qdot), limits, atol=1e-7, rtol=1e-5),
        bool(fit.success),
        str(fit.message),
    )
