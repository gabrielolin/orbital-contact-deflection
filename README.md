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

## Structured model-based contact interface

`ContactActionDecoder` accepts a `DecoderState`, an estimator `InterceptCorridor`,
and eight normalized coordinates. It returns the requested contact goal, Mink
terminal IK, bounded twist mapping, and trajectory-backend diagnostics.

| Coordinate | Meaning |
|---|---|
| 0 | Coupled XYZ/time along one reachable mean-path interval |
| 1–2 | Shield-normal tilt about contact tangent axes |
| 3–5 | EE linear velocity: normal, tangent1, tangent2 |
| 6–7 | Angular velocity about tangent1/tangent2; zero scales by default |

The conventions and affine scales are defined in
`control/contact_action_decoder.py` and `configs/decoder.yaml`. The contact
frame is `[normal,tangent1,tangent2]`; the MJCF shield's normal is local **+z**.
World-axis tilt rotations are left-multiplied onto the nominal shield rotation.
The decoder keeps the previous tangent frame, absolute selected time, and IK
seed for continuity. Call `decoder.reset()` at episode reset. Supply a belief
propagated to the current state timestamp. Stationary mean paths retain the
previous frame, or use an identity contact frame initially.

Build a workspace once and export its accepted samples as CSV:

```bash
python scripts/build_reachable_workspace.py --config configs/decoder.yaml
python scripts/decode_contact.py --nominal-line
```

The second command uses an explicitly synthetic KF mean through the nominal
shield position. Without `--nominal-line`, it uses the environment's current
filter and may report a normal no-intersection result. A small reproducible
smoke run is:

```bash
python scripts/build_reachable_workspace.py --samples 20 --output scratch/workspace-smoke.npz
python scripts/decode_contact.py --workspace scratch/workspace-smoke.npz --nominal-line
```

Cache reuse checks a fingerprint of model, IK/sampling settings, seed, and
initial configuration. A settings mismatch requires a new output path or
explicit cache replacement. Samples are stored in spacecraft coordinates;
`workspace.placed(base_position, base_rotation)` maps them into the current
world frame. Normal environment startup does not build or load a cache.

Workspace membership is a small union of balls around accepted constrained-IK
samples. Only the centers are validated: neighborhoods and unsampled regions
are approximate, and containment is not a guarantee for every orientation.
Sampling leaves orientation uncosted; terminal IK uses a weighted full
`FrameTask`. Robot self/spacecraft collision constraints exclude the projectile
and Mink's welded/adjacent body pairs. The imported model omits joint ranges;
configured ±2π planning bounds apply there and are not verified hardware limits.

IK and terminal twist mapping freeze the actual current base pose. These are
kinematic predictions, without floating-base momentum/trajectory prediction.
Requested position and time always remain on the KF mean line even when IK or
twist matching fails. No interval shrinking, action resampling, or dynamic
feasibility rejection is performed. No-intersection is a typed normal result.

`JointTrajectoryGenerator` is an injectable protocol. The default placeholder
returns `trajectory=None`, `status="not_implemented"`, and NaN residuals meaning
unknown. It does not yet generate or certify jerk-limited motion. A future
QP/SQP backend will own this work. Mink uses its bundled DAQP backend only for
IK; no trajectory solver or RL package is added. The existing six-joint-target
Gymnasium `step` API remains the execution smoke baseline; the Box(8) decoder
is a separate model-based interface pending an executable trajectory backend.
