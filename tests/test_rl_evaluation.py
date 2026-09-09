"""Deterministic evaluation summaries and checkpoint discovery."""

import os
from dataclasses import replace

import numpy as np
import pytest

from contact_deflection.envs.contact_deflection_env import ContactDeflectionConfig
from contact_deflection.rl.evaluation import evaluate_policy, latest_checkpoint
from contact_deflection.rl.train import SACTrainConfig


class ZeroPolicy:
    def predict(
        self, observation: dict[str, np.ndarray], *, deterministic: bool
    ) -> tuple[np.ndarray, None]:
        assert deterministic
        assert set(observation) == {"observation", "desired_goal"}
        return np.zeros(8), None


def test_sac_uses_evidence_based_adaptive_entropy_default() -> None:
    config = SACTrainConfig()
    assert config.entropy_coefficient == "auto_0.3"
    assert config.progress_bar
    assert not config.wandb_enabled


def test_evaluation_records_seeded_episode_and_aggregate_metrics() -> None:
    config = replace(ContactDeflectionConfig(), episode_duration=0.25)
    result = evaluate_policy(ZeroPolicy(), config, [31, 32])

    assert result["aggregate"]["episodes"] == 2
    assert result["aggregate"]["contact_rate"] == 0.0
    assert [episode["seed"] for episode in result["episodes"]] == [31, 32]
    assert all(episode["terminal_reason"] == "miss" for episode in result["episodes"])
    assert all(episode["velocity_error"] is None for episode in result["episodes"])


def test_latest_checkpoint_uses_recursive_modification_time(tmp_path) -> None:
    older = tmp_path / "best" / "best_model.zip"
    newer = tmp_path / "final_model.zip"
    older.parent.mkdir()
    older.write_bytes(b"older")
    newer.write_bytes(b"newer")
    os.utime(older, ns=(1, 1))
    os.utime(newer, ns=(2, 2))

    assert latest_checkpoint(tmp_path) == newer


def test_latest_checkpoint_requires_a_saved_model(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="no .zip checkpoints"):
        latest_checkpoint(tmp_path)
