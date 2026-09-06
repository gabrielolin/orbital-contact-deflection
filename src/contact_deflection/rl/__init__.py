"""RL experiment integration for the structured contact benchmark."""

from contact_deflection.rl.train import SACTrainConfig, ensure_workspace, train_sac

__all__ = ["SACTrainConfig", "ensure_workspace", "train_sac"]
