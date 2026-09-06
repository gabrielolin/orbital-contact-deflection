"""Typed YAML settings for the standalone model-based decoder workflow."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from contact_deflection.control.contact_action_decoder import DecoderConfig
from contact_deflection.control.trajectory import JointTrajectoryLimits
from contact_deflection.kinematics.mink_ik import IKConfig, MinkIK
from contact_deflection.kinematics.reachable_workspace import WorkspaceConfig


@dataclass(frozen=True)
class TrajectoryConfig:
    acceleration_limits: tuple[float, ...] = (5.0,) * 6
    jerk_limits: tuple[float, ...] = (30.0,) * 6

    def limits(self, ik: MinkIK) -> JointTrajectoryLimits:
        return JointTrajectoryLimits(
            ik.joint_lower,
            ik.joint_upper,
            ik.velocity_limits,
            self.acceleration_limits,
            self.jerk_limits,
        )


@dataclass(frozen=True)
class StructuredDecoderConfig:
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    ik: IKConfig = field(default_factory=IKConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    prediction_horizon: float = 5.0

    @classmethod
    def load(cls, path: str | Path) -> "StructuredDecoderConfig":
        with Path(path).open() as stream:
            values = yaml.safe_load(stream)
        return cls(
            workspace=WorkspaceConfig(**values.get("workspace", {})),
            ik=IKConfig(**values.get("ik", {})),
            decoder=DecoderConfig(**values.get("decoder", {})),
            trajectory=TrajectoryConfig(**values.get("trajectory", {})),
            prediction_horizon=values.get("prediction_horizon", 5.0),
        )
