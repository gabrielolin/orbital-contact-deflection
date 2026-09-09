#!/usr/bin/env python
"""Train SB3 SAC through the structured decoder and torque-control stack."""

import argparse
from pathlib import Path

from contact_deflection.control.decoder_config import StructuredDecoderConfig
from contact_deflection.envs.contact_deflection_env import ContactDeflectionConfig
from contact_deflection.rl.train import SACTrainConfig, train_sac


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder-config", default="configs/decoder.yaml")
    parser.add_argument("--env-config", default="configs/env.yaml")
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-dir", default="outputs/sac")
    arguments = parser.parse_args()
    decoder = StructuredDecoderConfig.load(arguments.decoder_config)
    task = ContactDeflectionConfig.load(arguments.env_config, decoder=decoder)
    model_path = train_sac(
        task,
        SACTrainConfig(
            total_timesteps=arguments.timesteps,
            seed=arguments.seed,
            run_dir=arguments.run_dir,
        ),
    )
    print(f"Saved SAC model: {Path(model_path)}")


if __name__ == "__main__":
    main()
