#!/usr/bin/env python
"""Train or evaluate SB3 SAC through the structured control stack."""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Full training commonly runs without an X server. This must be set before
# importing MuJoCo through the environment package.
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from stable_baselines3 import SAC

from contact_deflection.control.decoder_config import StructuredDecoderConfig
from contact_deflection.envs.contact_deflection_env import ContactDeflectionConfig
from contact_deflection.rl import (
    SACTrainConfig,
    evaluate_policy,
    latest_checkpoint,
    render_policy_episode,
    train_sac,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _parse_entropy_coefficient(value: str) -> str | float:
    if value == "auto" or value.startswith("auto_"):
        return value
    try:
        coefficient = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "entropy coefficient must be a positive number, auto, or auto_<initial>"
        ) from error
    if coefficient <= 0:
        raise argparse.ArgumentTypeError("entropy coefficient must be positive")
    return coefficient


def _entropy_summary(model: SAC) -> dict[str, str | float]:
    result: dict[str, str | float] = {"configured": str(model.ent_coef)}
    if model.log_ent_coef is not None:
        result["current"] = float(model.log_ent_coef.detach().exp().cpu().item())
    else:
        result["current"] = float(model.ent_coef)
    return result


def _start_wandb(
    arguments: argparse.Namespace,
    task: ContactDeflectionConfig,
) -> Any | None:
    if arguments.wandb_project is None:
        return None
    tracking_dir = arguments.run_dir / "wandb"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    # Keep all W&B state with the experiment. In particular, artifact staging
    # otherwise defaults to a user-level directory that may be read-only on
    # managed compute nodes.
    os.environ.setdefault("WANDB_DATA_DIR", str(tracking_dir / "data"))
    os.environ.setdefault("WANDB_CACHE_DIR", str(tracking_dir / "cache"))
    try:
        import wandb
    except ImportError as error:  # pragma: no cover - installation error path
        raise ImportError(
            "Install tracking support with `pip install -e '.[tracking]'`."
        ) from error
    return wandb.init(
        project=arguments.wandb_project,
        entity=arguments.wandb_entity,
        name=arguments.wandb_name,
        tags=arguments.wandb_tags,
        mode=arguments.wandb_mode,
        dir=str(tracking_dir),
        config={
            "mode": "eval_only" if arguments.eval_only else "training",
            "timesteps": arguments.timesteps,
            "seed": arguments.seed,
            "num_envs": arguments.num_envs,
            "entropy_coefficient": arguments.ent_coef,
            "evaluation_episodes": arguments.eval_episodes,
            "render_episodes": arguments.render_episodes,
            "evaluation_seed": arguments.eval_seed,
            "post_contact_seconds": arguments.post_contact_seconds,
            "task": _jsonable(asdict(task)),
        },
        sync_tensorboard=not arguments.eval_only,
        save_code=False,
    )


def _log_wandb_outputs(run: Any, summary_path: Path, checkpoint: Path) -> None:
    import wandb

    summary = json.loads(summary_path.read_text())
    aggregate = summary["evaluation"]["aggregate"]
    metrics = {
        f"final_evaluation/{key}": value
        for key, value in aggregate.items()
        if value is not None
    }
    videos = {
        f"final_evaluation/video_{index:02d}": wandb.Video(
            rollout["video"], fps=30, format="mp4"
        )
        for index, rollout in enumerate(summary["rendered_episodes"])
    }
    run.log({**metrics, **videos})
    for key, value in metrics.items():
        run.summary[key] = value
    artifact = wandb.Artifact("contact-deflection-policy", type="model")
    artifact.add_file(str(checkpoint), name=checkpoint.name)
    artifact.add_file(str(summary_path), name=summary_path.name)
    run.log_artifact(artifact)


def _evaluate_and_render(
    model: SAC,
    task: ContactDeflectionConfig,
    checkpoint: Path,
    run_dir: Path,
    *,
    evaluation_episodes: int,
    render_episodes: int,
    evaluation_seed: int,
    post_contact_seconds: float,
    mode: str,
    training_timesteps: int | None,
    training_seed: int | None,
    training_num_envs: int | None,
) -> Path:
    if evaluation_episodes < 1 or not 0 <= render_episodes <= evaluation_episodes:
        raise ValueError(
            "evaluation episodes must be positive and render episodes must be "
            "between zero and the evaluation count"
        )
    if post_contact_seconds < 0:
        raise ValueError("post-contact duration must be nonnegative")
    seeds = list(range(evaluation_seed, evaluation_seed + evaluation_episodes))
    evaluation = evaluate_policy(model, task, seeds)
    render_dir = run_dir / (
        "renders" if mode == "post_training" else "eval_only_renders"
    )
    rendered = [
        render_policy_episode(
            model,
            task,
            render_dir / f"episode_{index:02d}_seed_{seed}.mp4",
            seed=seed,
            post_contact_seconds=post_contact_seconds,
        )
        for index, seed in enumerate(seeds[:render_episodes])
    ]
    summary = {
        "mode": mode,
        "checkpoint": str(checkpoint.resolve()),
        "training_timesteps": training_timesteps,
        "training_seed": training_seed,
        "training_num_envs": training_num_envs,
        "evaluation_seeds": seeds,
        "evaluation": evaluation,
        "rendered_episodes": rendered,
        "entropy_coefficient": _entropy_summary(model),
        "resolved_task_config": _jsonable(asdict(task)),
    }
    summary_path = run_dir / (
        "evaluation_summary.json"
        if mode == "post_training"
        else "eval_only_summary.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    aggregate = evaluation["aggregate"]
    print(
        "Evaluation: "
        f"return={aggregate['return_mean']:.3f}±{aggregate['return_std']:.3f}, "
        f"contact={aggregate['contact_rate']:.3f}, "
        f"velocity_error={aggregate['velocity_error_mean']}"
    )
    print(f"Evaluation summary: {summary_path}")
    for rollout in rendered:
        print(f"Rendered episode: {rollout['video']}")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder-config", default="configs/decoder.yaml")
    parser.add_argument("--env-config", default="configs/env.yaml")
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="parallel rollout environments (default: 8; use 1 for serial)",
    )
    parser.add_argument(
        "--ent-coef",
        type=_parse_entropy_coefficient,
        default="auto_0.3",
        help="SAC entropy coefficient (default: adaptive, initialized at 0.3)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_false",
        dest="progress_bar",
        help="disable the tqdm training progress bar",
    )
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/sac"))
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="skip training and evaluate the latest checkpoint under --run-dir",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="explicit checkpoint for --eval-only (defaults to most recent)",
    )
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--render-episodes", type=int, default=3)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--post-contact-seconds", type=float, default=4.0)
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", "contact-deflection"),
        help=(
            "W&B project name (default: contact-deflection; WANDB_PROJECT "
            "overrides the default)"
        ),
    )
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-name")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("WANDB_MODE", "online"),
    )
    parser.add_argument(
        "--wandb-tag",
        action="append",
        dest="wandb_tags",
        default=[],
        help="repeat to attach multiple W&B tags",
    )
    arguments = parser.parse_args()
    if arguments.checkpoint is not None and not arguments.eval_only:
        parser.error("--checkpoint is only valid with --eval-only")
    if arguments.num_envs < 1:
        parser.error("--num-envs must be positive")

    decoder = StructuredDecoderConfig.load(arguments.decoder_config)
    task = ContactDeflectionConfig.load(arguments.env_config, decoder=decoder)
    arguments.run_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = _start_wandb(arguments, task)
    try:
        if arguments.eval_only:
            checkpoint = (
                arguments.checkpoint
                if arguments.checkpoint is not None
                else latest_checkpoint(arguments.run_dir)
            )
            if not checkpoint.is_file():
                parser.error(f"checkpoint does not exist: {checkpoint}")
            model = SAC.load(checkpoint)
            print(f"Loaded SAC checkpoint: {checkpoint}")
            summary_path = _evaluate_and_render(
                model,
                task,
                checkpoint,
                arguments.run_dir,
                evaluation_episodes=arguments.eval_episodes,
                render_episodes=arguments.render_episodes,
                evaluation_seed=arguments.eval_seed,
                post_contact_seconds=arguments.post_contact_seconds,
                mode="eval_only",
                training_timesteps=None,
                training_seed=None,
                training_num_envs=None,
            )
        else:
            checkpoint = train_sac(
                task,
                SACTrainConfig(
                    total_timesteps=arguments.timesteps,
                    seed=arguments.seed,
                    run_dir=str(arguments.run_dir),
                    entropy_coefficient=arguments.ent_coef,
                    progress_bar=arguments.progress_bar,
                    wandb_enabled=wandb_run is not None,
                    num_envs=arguments.num_envs,
                ),
            )
            print(f"Saved SAC checkpoint: {checkpoint}")
            model = SAC.load(checkpoint)
            summary_path = _evaluate_and_render(
                model,
                task,
                checkpoint,
                arguments.run_dir,
                evaluation_episodes=arguments.eval_episodes,
                render_episodes=arguments.render_episodes,
                evaluation_seed=arguments.eval_seed,
                post_contact_seconds=arguments.post_contact_seconds,
                mode="post_training",
                training_timesteps=arguments.timesteps,
                training_seed=arguments.seed,
                training_num_envs=arguments.num_envs,
            )
        if wandb_run is not None:
            _log_wandb_outputs(wandb_run, summary_path, checkpoint)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
