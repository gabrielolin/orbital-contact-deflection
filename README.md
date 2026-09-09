# Contact Deflection

Research code for contact-aware projectile deflection by a free-floating
six-joint UR5 manipulator. The environment is a modern MuJoCo/Gymnasium port of
the simple SpaceRobotEnv design, extended with an independently moving
projectile, a torque-level PD controller, and noisy spacecraft-frame sensing.
SAC policies use a structured contact-intent action rather than raw joint targets.

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
python scripts/visualize_reachable_workspace.py --config configs/decoder.yaml
```

## Experiments

Train the initial SB3 SAC baseline:

```bash
python experiments/train_sac.py --timesteps 10000 --seed 0
```

## Repository Structure

- `assets/mjcf/`: free-floating spacecraft, UR5 arm, shield, and projectile scene.
- `third_party/`: pinned SpaceRobotEnv meshes and Apache-2.0 license.
- `src/contact_deflection/envs/`: SpaceRobotEnv foundation and deflection task.
- `src/contact_deflection/estimation/`: world-state projectile Kalman filter.
- `src/contact_deflection/control/`: decoder, kinematic reference, and torque control.
- `src/contact_deflection/rl/`: thin Stable-Baselines3 training orchestration.
- `src/contact_deflection/visualization/`: headless frame/video rendering.
- `experiments/`: training and evaluation orchestration.
- `scripts/`: durable operational utilities such as workspace visualization.
- `configs/`: explicit simulation and controller parameters.
- `tests/`: deterministic analytical and simulation checks.

The arm geometry, inertial chain, and environment structure are adapted from
SpaceRobotEnv's UR5 model
at the commit recorded in `THIRD_PARTY.md`. The spacecraft uses a simplified
1.2 m cuboid bus with visual-only, zero-mass solar-panel wings rather than
flight-qualified CAD. The UR5 is mounted flush to the bus's +Y front face and
starts in a folded, outward-facing home posture. The bus retains the original
200 kg mass and inertia.
The projectile is unforced before contact and therefore moves at constant
velocity in the MuJoCo world frame. Its noisy measurements and policy belief
are expressed relative to the moving spacecraft.

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

Render the configured smooth workspace envelope as a PNG from the reverse
overview camera:

```bash
python scripts/visualize_reachable_workspace.py --config configs/decoder.yaml
```

Override the image path with `--visualization path/to/workspace.png`. The
ellipsoid center and radii are configured in spacecraft coordinates in
`configs/decoder.yaml`; it is placed at the current base pose before line
intersection. For a near-normal projectile, the Y radius is the principal knob
for contact-corridor length. The smooth envelope defines the continuous action
chart but is not itself a reachability certificate. Every action-selected pose
and orientation is checked online with Mink constrained IK; an unsuccessful IK
proposal is exposed in step diagnostics, incurs the residual penalty, and does
not drive a best-effort trajectory. Robot
self/spacecraft collision constraints exclude the projectile and Mink's
welded/adjacent body pairs. The imported model omits joint ranges; configured
±2π planning bounds apply there and are not verified hardware limits.

IK and terminal twist mapping freeze the actual current base pose. These are
kinematic predictions, without floating-base momentum/trajectory prediction.
Requested position and time always remain on the KF mean line even when IK or
twist matching fails. No interval shrinking, action resampling, or dynamic
feasibility rejection is performed. No-intersection is a typed normal result.

`JointTrajectoryGenerator` is an injectable protocol. The executable baseline
uses endpoint-quintic references and samples their position, velocity, and
acceleration bounds for diagnostics. It never clips a violating trajectory, so
`limit_violation` is not a feasibility certificate. A future QP/SQP backend can
replace it behind the same protocol. Mink uses DAQP only for IK.

`ContactDeflectionEnv` is the canonical Box(8) Gymnasium environment. Every
policy action is decoded into a KF mean-line contact goal, constrained IK,
bounded terminal twist, quintic joint reference, and PD torque command. It
constructs its smooth base-frame workspace directly from configuration; no
offline reachability cache is required. The environment returns a dictionary
with a 49D `observation` vector and a 3D `desired_goal` outgoing-projectile
velocity expressed in the current spacecraft axes. SAC uses SB3's
`MultiInputPolicy`. Each policy action controls a 250 ms trajectory prefix,
while torque references update every 5 ms and MuJoCo integrates at 1 ms.
Rewards are zero before termination and score outgoing-velocity error, miss
closest approach, and spacecraft angular-momentum transfer at termination.
Every seeded reset samples the initial projectile line, arm state, spacecraft
rates, projectile and shield masses, and stable MuJoCo contact parameters from
the Gaussian distributions in `configs/env.yaml`. The resolved episode values
are returned in `info["episode_parameters"]` for reproducibility.
`experiments/train_sac.py` records model/evaluation outputs under `outputs/`.
