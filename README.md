# Contact Deflection

Research code for contact-aware projectile deflection by a free-floating
six-joint UR5 manipulator. The environment is a modern MuJoCo/Gymnasium port of
the simple SpaceRobotEnv design, extended with an independently moving
projectile, a joint-space PD controller, and noisy spacecraft-frame sensing.
Learning methods are intentionally not implemented yet.

The canonical environment is registered as `ContactDeflection-v0`:

```python
import gymnasium as gym
import contact_deflection

env = gym.make("ContactDeflection-v0")
observation, info = env.reset(seed=0)
```

## Installation

Create the Conda environment from the repository root:

```bash
conda env create -f environment.yml
conda activate contact-deflection
```

The editable package is installed by the environment file. To update an
existing environment, run `conda env update -f environment.yml --prune`.

## Quick Start

```bash
python scripts/sanity_check_env.py --steps 100 --seed 0
```

## Experiments

There is no training experiment in this milestone. Produce a deterministic
headless MP4 smoke render with:

```bash
python scripts/render_episode.py --steps 120 --seed 0 --output videos/smoke.mp4
```

## Repository Structure

- `assets/mjcf/`: free-floating spacecraft, UR5 arm, shield, and projectile scene.
- `third_party/`: pinned SpaceRobotEnv meshes and Apache-2.0 license.
- `src/contact_deflection/envs/`: SpaceRobotEnv foundation and deflection task.
- `src/contact_deflection/estimation/`: world-state projectile Kalman filter.
- `src/contact_deflection/control/`: basic torque-level PD control.
- `src/contact_deflection/visualization/`: headless frame/video rendering.
- `configs/`: explicit simulation and controller parameters.
- `tests/`: deterministic analytical and simulation checks.

The arm geometry, inertial chain, and environment structure are adapted from
SpaceRobotEnv's UR5 model
at the commit recorded in `THIRD_PARTY.md`. The spacecraft bus remains the
upstream cuboid approximation rather than a flight-qualified CAD model. The
projectile is unforced before contact and therefore moves at constant velocity
in the MuJoCo world frame. Its noisy measurements and policy belief are
expressed relative to the moving spacecraft.

The estimator propagates projectile position and velocity in world coordinates
with a constant-velocity model. At each sensor update, the current MuJoCo
spacecraft pose defines a time-varying linear measurement matrix. Spacecraft
state is treated as known; estimating uncertain spacecraft localization would
require a later error-state EKF.
