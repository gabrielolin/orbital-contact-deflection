"""Contact-aware projectile-deflection research package."""

import gymnasium as gym

from contact_deflection.envs import ContactDeflectionEnv

if "ContactDeflection-v0" not in gym.registry:
    gym.register(
        id="ContactDeflection-v0",
        entry_point="contact_deflection.envs:ContactDeflectionEnv",
    )

__all__ = ["ContactDeflectionEnv"]
