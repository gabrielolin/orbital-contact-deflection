"""Mink IK restricted to arm DOFs at a measured, frozen spacecraft pose.

No environment data or model limits are modified. Differential IK QPs are
restricted algebraically to arm columns, locking base and projectile exactly.
The upstream UR5 MJCF omits joint limits: configurable +/-2 pi fallback bounds
are planning assumptions, not a claim of model-enforced hardware limits.
"""

from dataclasses import dataclass
from typing import Any

import mink
import mujoco
import numpy as np
import qpsolvers

from contact_deflection.envs.space_robot_env import SpaceRobotEnv


@dataclass(frozen=True)
class IKConfig:
    max_iterations: int = 100
    iteration_dt: float = 0.05
    position_cost: float = 10.0
    orientation_cost: float = 1.0
    position_tolerance: float = 0.003
    orientation_tolerance: float = 0.03
    posture_cost: float = 1e-4
    damping: float = 1e-5
    velocity_limits: tuple[float, ...] = (2.0,) * 6
    joint_lower: tuple[float, ...] = (-2 * np.pi,) * 6
    joint_upper: tuple[float, ...] = (2 * np.pi,) * 6
    collision_enabled: bool = True
    collision_clearance: float = 0.002
    collision_detection_distance: float = 0.05
    solver: str = "daqp"


@dataclass(frozen=True)
class IKSolution:
    q_target: np.ndarray
    position_residual: float
    orientation_residual: float
    converged: bool
    metadata: dict[str, Any]


class MinkIK:
    """Best-effort pose IK; robot-pose prediction during motion is downstream."""

    def __init__(self, model: mujoco.MjModel, config: IKConfig | None = None):
        self.model = model
        self.config = config or IKConfig()
        cfg = self.config
        ids = np.array([model.joint(n).id for n in SpaceRobotEnv.ARM_JOINT_NAMES])
        self.arm_qpos_indices = model.jnt_qposadr[ids].copy()
        self.arm_dof_indices = model.jnt_dofadr[ids].copy()
        limited = model.jnt_limited[ids].astype(bool)
        self.joint_lower = np.where(limited, model.jnt_range[ids, 0], cfg.joint_lower)
        self.joint_upper = np.where(limited, model.jnt_range[ids, 1], cfg.joint_upper)
        self.velocity_limits = np.asarray(cfg.velocity_limits, dtype=float)
        if (
            self.velocity_limits.shape != (6,)
            or self.joint_lower.shape != (6,)
            or self.joint_upper.shape != (6,)
            or np.any(self.joint_lower >= self.joint_upper)
            or np.any(self.velocity_limits <= 0)
            or cfg.max_iterations < 1
            or cfg.iteration_dt <= 0
            or cfg.position_cost <= 0
            or cfg.orientation_cost < 0
            or cfg.position_tolerance <= 0
            or cfg.orientation_tolerance <= 0
            or cfg.posture_cost < 0
            or cfg.damping <= 0
            or cfg.collision_clearance < 0
            or cfg.collision_detection_distance <= cfg.collision_clearance
            or not all(
                np.all(np.isfinite(x))
                for x in (self.velocity_limits, self.joint_lower, self.joint_upper)
            )
        ):
            raise ValueError("Invalid IK bounds, costs, or tolerances")
        self.configuration = mink.Configuration(model)
        self.site_id = model.site("shield_center").id
        self._uses_fallback_limits = bool(np.any(~limited))
        # Mink removes welded and adjacent-body pairs. Projectile geometry is
        # intentionally excluded: this is a terminal *contact* pose solver.
        geoms = [
            model.geom(i).name
            for i in range(model.ngeom)
            if model.geom(i).name != "projectile_geom"
            and model.geom(i).group[0] != 2
        ]
        self._collision_limit = mink.CollisionAvoidanceLimit(
            model,
            [(geoms, geoms)],
            minimum_distance_from_collisions=cfg.collision_clearance,
            collision_detection_distance=cfg.collision_detection_distance,
        )

    def _update(self, full_qpos: np.ndarray, q_target: np.ndarray | None) -> None:
        q = np.asarray(full_qpos, dtype=float).copy()
        if q.shape != (self.model.nq,) or not np.all(np.isfinite(q)):
            raise ValueError("full_qpos must be a finite model-sized configuration")
        if q_target is not None:
            arm = np.asarray(q_target, dtype=float)
            if arm.shape != (6,) or not np.all(np.isfinite(arm)):
                raise ValueError("Arm configuration must be a finite six-vector")
            q[self.arm_qpos_indices] = arm
        self.configuration.update(q)

    def pose(
        self,
        full_qpos: np.ndarray,
        q_target: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._update(full_qpos, q_target)
        data = self.configuration.data
        return (
            data.site_xpos[self.site_id].copy(),
            data.site_xmat[self.site_id].reshape(3, 3).copy(),
        )

    def jacobian(self, full_qpos: np.ndarray, q_target: np.ndarray) -> np.ndarray:
        """Return world [linear; angular] Jacobian with the base held fixed."""
        self._update(full_qpos, q_target)
        linear = np.zeros((3, self.model.nv))
        angular = np.zeros_like(linear)
        mujoco.mj_jacSite(
            self.model, self.configuration.data, linear, angular, self.site_id
        )
        return np.vstack((linear, angular))[:, self.arm_dof_indices]

    def _constraints(self) -> tuple[bool, float]:
        q = self.configuration.q[self.arm_qpos_indices]
        valid = bool(
            np.all(q >= self.joint_lower - 1e-8)
            and np.all(q <= self.joint_upper + 1e-8)
        )
        minimum = float("inf")
        if self.config.collision_enabled:
            for first, second in self._collision_limit.geom_id_pairs:
                distance = mujoco.mj_geomDistance(
                    self.model,
                    self.configuration.data,
                    first,
                    second,
                    self.config.collision_detection_distance,
                    None,
                )
                minimum = min(minimum, float(distance))
            valid = valid and minimum >= self.config.collision_clearance - 1e-7
        return valid, minimum

    def solve(
        self,
        full_qpos: np.ndarray,
        position_W: np.ndarray,
        orientation_W: np.ndarray,
        seed: np.ndarray | None = None,
    ) -> IKSolution:
        cfg = self.config
        position = np.asarray(position_W, dtype=float)
        orientation = np.asarray(orientation_W, dtype=float)
        if (
            position.shape != (3,)
            or orientation.shape != (3, 3)
            or not np.all(np.isfinite(position))
            or not np.all(np.isfinite(orientation))
            or not np.allclose(orientation.T @ orientation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(orientation), 1.0, atol=1e-6)
        ):
            raise ValueError("Expected finite XYZ and proper rotation matrix")
        self._update(full_qpos, seed)
        initial = self.configuration.q.copy()
        # An invalid warm start is repaired explicitly; requested task is untouched.
        initial[self.arm_qpos_indices] = np.clip(
            initial[self.arm_qpos_indices], self.joint_lower, self.joint_upper
        )
        self.configuration.update(initial)
        frame = mink.FrameTask(
            "shield_center",
            "site",
            cfg.position_cost,
            cfg.orientation_cost,
            lm_damping=cfg.damping,
        )
        frame.set_target(
            mink.SE3.from_rotation_and_translation(
                mink.SO3.from_matrix(orientation), position
            )
        )
        posture = mink.PostureTask(self.model, cost=cfg.posture_cost)
        posture.set_target(initial)
        limits = [self._collision_limit] if cfg.collision_enabled else []
        best: (
            tuple[tuple[bool, float], np.ndarray, float, float, bool, float] | None
        ) = None
        status = "iteration_limit"
        iteration = 0
        for iteration in range(cfg.max_iterations + 1):
            data = self.configuration.data
            pos_error = float(np.linalg.norm(data.site_xpos[self.site_id] - position))
            actual = data.site_xmat[self.site_id].reshape(3, 3)
            rot_error = float(
                np.linalg.norm(mink.SO3.from_matrix(orientation.T @ actual).log())
            )
            valid, clearance = self._constraints()
            score = (
                not valid,
                cfg.position_cost * pos_error + cfg.orientation_cost * rot_error,
            )
            if best is None or score < best[0]:
                best = (
                    score,
                    self.configuration.q[self.arm_qpos_indices].copy(),
                    pos_error,
                    rot_error,
                    valid,
                    clearance,
                )
            if (
                valid
                and pos_error <= cfg.position_tolerance
                and (
                    cfg.orientation_cost == 0 or rot_error <= cfg.orientation_tolerance
                )
            ):
                best = (
                    score,
                    self.configuration.q[self.arm_qpos_indices].copy(),
                    pos_error,
                    rot_error,
                    valid,
                    clearance,
                )
                status = "converged"
                break
            if iteration == cfg.max_iterations:
                break
            # Unit pseudo-time makes Mink's displacement/velocity collision
            # inequalities identical; physical IK step limits are explicit below.
            problem = mink.build_ik(
                self.configuration,
                [frame, posture],
                1.0,
                damping=cfg.damping,
                limits=limits,
            )
            indices = self.arm_dof_indices
            hessian = np.asarray(problem.P)[np.ix_(indices, indices)]
            gradient = np.asarray(problem.q)[indices]
            matrix = None if problem.G is None else np.asarray(problem.G)[:, indices]
            bound = None if problem.h is None else np.asarray(problem.h)
            if bound is not None:
                finite = np.isfinite(bound)
                assert matrix is not None
                matrix, bound = matrix[finite], bound[finite]
            q = self.configuration.q[self.arm_qpos_indices]
            step_limit = self.velocity_limits * cfg.iteration_dt
            try:
                delta = qpsolvers.solve_qp(
                    hessian,
                    gradient,
                    matrix,
                    bound,
                    lb=np.maximum(self.joint_lower - q, -step_limit),
                    ub=np.minimum(self.joint_upper - q, step_limit),
                    solver=cfg.solver,
                )
            except (ValueError, RuntimeError, qpsolvers.exceptions.QPError) as error:
                status = f"qp_failure: {error}"
                break
            if delta is None or not np.all(np.isfinite(delta)):
                status = "qp_no_solution"
                break
            updated = self.configuration.q.copy()
            updated[self.arm_qpos_indices] += delta
            self.configuration.update(updated)
        assert best is not None
        _, target, pos_error, rot_error, valid, clearance = best
        return IKSolution(
            target,
            pos_error,
            rot_error,
            status == "converged",
            {
                "status": status,
                "iterations": iteration,
                "constraints_satisfied": valid,
                "minimum_collision_distance": clearance,
                "collision_enabled": cfg.collision_enabled,
                "collision_pairs": len(self._collision_limit.geom_id_pairs),
                "fallback_joint_limits": self._uses_fallback_limits,
                "base_model": "frozen_current_pose",
            },
        )
