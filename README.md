# Contact Deflection

Contact Deflection studies goal-conditioned projectile redirection with a
free-floating spacecraft manipulator. A six-joint UR5 carries a square shield;
an incoming projectile is observed through noisy spacecraft-frame position
measurements; moving the arm reacts on the uncontrolled spacecraft base.

The scientific question is:

> Can a low-frequency learned policy choose *where, when, and how* to contact a
> projectile while a model-based stack realizes those decisions at control and
> simulation rates—and can it direct the outgoing velocity without excessive
> spacecraft angular-momentum disturbance?

The intended comparison unit is a complete approach mapping the benchmark's
belief and desired outgoing velocity to its structured contact action. The
current approach is Soft Actor-Critic (SAC) above a fixed model-based decoder.
This isolates learning of contact strategy from learning joint-level robot
motion.

## Current status

The repository contains an executable MuJoCo/Gymnasium benchmark, Gaussian
projectile state estimator, structured end-effector contact decoder,
collision-aware terminal IK, bounded terminal-twist fitting, quintic joint
references, torque control, SB3 SAC training, deterministic evaluation, and
video rendering. It is research code, not a flight-dynamics or hardware-fidelity
simulator; important modeling limitations are listed below.

## Scientific formulation

### Randomized partially observed MDP

An episode samples physical and initial-state parameters

$$
\phi \sim p(\phi),
$$

including projectile and shield mass, MuJoCo contact compliance/damping, arm
configuration, spacecraft rates, and projectile position and velocity. The
underlying simulator state contains spacecraft pose and twist, arm joint state,
and projectile state:

$$
x_t = \left(T^W_{B,t},\,V^W_{B,t},\,q_t,\,\dot q_t,\,p^W_t,\,v^W_t\right).
$$

MuJoCo supplies the nonlinear free-floating transition, including arm/base
reaction and contact:

$$
x_{k+1}=F_{\Delta t_{\mathrm{sim}}}(x_k,u_k;\phi),
\qquad \Delta t_{\mathrm{sim}}=1\ \mathrm{ms}.
$$

The policy does not receive ground-truth projectile state or contact time. Its
raw measurement is noisy projectile position in spacecraft coordinates:

$$
z_k = (R^W_{B,k})^\top\left(p^W_k-r^W_{B,k}\right)+\epsilon_k,
\qquad \epsilon_k\sim\mathcal N(0,\Sigma_z).
$$

A Kalman filter maintains a Gaussian belief over world-frame projectile
position and velocity using the constant-velocity prior

$$
\begin{bmatrix}p^W_{k+1}\\v^W_{k+1}\end{bmatrix}
=
\begin{bmatrix}I&\Delta t I\\0&I\end{bmatrix}
\begin{bmatrix}p^W_k\\v^W_k\end{bmatrix}+G w_k,
\qquad w_k\sim\mathcal N(0,\sigma_a^2 I),
$$

and a time-varying measurement matrix defined by the known spacecraft pose.
The posterior is transformed to spacecraft-relative coordinates before it is
given to the policy.

The Gymnasium observation is a dictionary suitable for SB3
`MultiInputPolicy`:

$$
o_t=\{\texttt{observation}\in\mathbb R^{49},\;
      \texttt{desired\_goal}\in\mathbb R^3\}.
$$

The 49-dimensional state vector contains relative belief mean and covariance
diagonal, arm position and velocity, spacecraft pose and twist, previous
action, current interception-corridor features, and episode progress. The goal
is the desired outgoing projectile velocity expressed in current spacecraft
axes.

### Temporal abstraction

The stochastic policy acts every 250 ms:

$$
a_t\sim\pi_\theta(a\mid o_t,g),\qquad a_t\in[-1,1]^8.
$$

Each action is decoded into a complete contact goal and joint trajectory. Only
the first 250 ms of that reference is executed before observing and replanning.
The induced policy-scale transition is therefore

$$
P_{250\mathrm{ms}}(x_{t+1}\mid x_t,a_t)
=\prod_{j=0}^{249}P_{1\mathrm{ms}}
\left(x_{t,j+1}\mid x_{t,j},u_{t,j}\right),
$$

with torque commands refreshed every 5 ms. This keeps contact simulation fine
while giving SAC a short receding-horizon decision sequence.

### Terminal objective

Rewards are zero before termination. After shield contact, the environment
executes a short follow-through and braking phase, waits for separation, and
averages the outgoing projectile velocity. The contact reward is

$$
r_T=
\exp\!\left[-\left(
\frac{\lVert v_{\mathrm{out}}-v_{\mathrm{goal}}\rVert_2}{\sigma_v}
\right)^2\right]
-w_L\,h_\delta\!\left(\frac{\lVert\Delta L_B\rVert_2}{L_0}\right),
$$

where $h_\delta$ is the Huber loss and $\Delta L_B$ is the change in angular
momentum of the spacecraft bus-and-arm subtree. If the projectile is missed,

$$
r_T=-c_{\mathrm{miss}}
-h_\delta\!\left(\frac{d_{\min}}{d_0}\right)
-w_L\,h_\delta\!\left(\frac{\lVert\Delta L_B\rVert_2}{L_0}\right).
$$

The benchmark terminates on contact after outcome measurement, or at the
configured episode deadline. All reward scales are in `configs/env.yaml`.

SAC optimizes the usual maximum-entropy objective

$$
J(\pi)=\mathbb E\!\left[
\sum_t\gamma^t\left(r_t+\alpha\,\mathcal H(\pi(\cdot\mid o_t,g))\right)
\right].
$$

The durable trainer uses adaptive entropy tuning initialized at
$\alpha_0=0.3$ (`auto_0.3`). In the truncated probes, fixed $\alpha=0.01$
under-explored and degraded held-out return, whereas automatic tuning improved
return and converged near $\alpha=0.325$. This is an evidence-based default,
not a completed hyperparameter study.

## Learning/model-based split

The policy chooses contact strategy; it does not output joint targets or
torques directly.

| Layer | Responsibility |
|---|---|
| Benchmark | Sample the randomized task, expose partial observations and goals, execute physics, and compute terminal metrics |
| SAC policy | Choose a normalized, end-effector-centric contact intent every 250 ms |
| Estimator | Infer a world-frame Gaussian projectile trajectory from noisy spacecraft-frame measurements |
| Decoder and IK | Convert latent intent to a reachable contact time, pose, and terminal shield twist |
| Trajectory and controller | Produce joint references and track them with bounded torque at 5 ms |
| MuJoCo | Integrate free-floating multibody and contact dynamics at 1 ms |

For projectile belief mean $(\hat p^W,\hat v^W)$, the decoder intersects the
mean spacetime line

$$
p^W(\tau)=\hat p^W+\tau\hat v^W
$$

with the spacecraft-attached ellipsoidal candidate workspace

$$
\mathcal W=\left\{p:\left\lVert
D^{-1}R_{WB}^\top(p-c^W)\right\rVert_2\le1\right\}.
$$

If the first candidate-workspace interval is $[\tau_-,\tau_+]$, action coordinate
$a_0$ selects

$$
\tau(a_0)=\tau_-+\frac{a_0+1}{2}(\tau_+-\tau_-).
$$

The full action semantics are:

| Coordinate | Decoded meaning |
|---|---|
| 0 | Coupled contact position and time along the estimated projectile line |
| 1–2 | Shield-normal tilt about the two contact tangent axes |
| 3–5 | Shield-center linear velocity in normal/tangent coordinates |
| 6–7 | Angular velocity about tangent axes; scales are currently zero by default |

The contact frame is $C=[n,t_1,t_2]$ with
$n=-\hat v^W/\lVert\hat v^W\rVert$. Mink IK constrains shield-center position
and alignment of the shield's local $+z$ normal. Local shield yaw is free, so
the wrist can use that redundancy to extend reach. IK freezes the measured
spacecraft pose and enforces configured joint and collision constraints.

At the terminal configuration, desired joint velocity solves a bounded,
regularized twist fit:

$$
\dot q^*=\arg\min_{-\dot q_{\max}\le\dot q\le\dot q_{\max}}
\left\lVert W^{1/2}(J(q)\dot q-\xi^*)\right\rVert_2^2
+\lambda\lVert\dot q\rVert_2^2.
$$

The executable trajectory backend then fits independent quintics

$$
q_j(t)=\sum_{i=0}^{5}c_{j,i}t^i
$$

to initial and requested terminal $(q,\dot q,\ddot q)$ at the fixed contact
time. This is endpoint interpolation, not trajectory optimization. Velocity,
acceleration, jerk, and position bounds are sampled for diagnostics, but a
reported `limit_violation` does not modify the curve. Torque tracking is

$$
u=\operatorname{clip}\left(
K_p(q_{\mathrm{ref}}-q)+K_d(\dot q_{\mathrm{ref}}-\dot q),
-u_{\max},u_{\max}\right).
$$

## Benchmark distribution

Every reset samples Gaussian dynamics and initial-state parameters from
`configs/env.yaml`. Projectile lines are centered on the configured workspace
and use broad transverse Gaussian variation. Outliers are rejection-sampled
unless the true future line crosses a 95%-scale inner workspace. This preserves
a truncated Gaussian without projecting probability mass onto the boundary and
ensures the policy receives a meaningful interception corridor. Episode
parameters are returned in `info["episode_parameters"]`.

The desired outgoing velocity is sampled in the incoming contact frame with a
positive normal component and Gaussian tangent components. Training therefore
asks the policy to preserve model-based interception while learning diverse
reflection directions under uncertain contact and inertial parameters.

## Installation

From the repository root:

```bash
conda env create -f environment.yml
conda activate contact-deflection
```

For an existing environment:

```bash
conda env update -f environment.yml --prune
conda activate contact-deflection
```

The environment installs the package editable and includes MuJoCo, Mink, SB3,
testing, and video dependencies. The equivalent pip development installation
is:

```bash
python -m pip install -e '.[dev,render,rl,tracking]'
```

## Running the code

Run the deterministic checks first:

```bash
pytest
ruff check src tests scripts experiments
mypy src
```

Inspect the smooth candidate workspace as a PNG:

```bash
python scripts/visualize_reachable_workspace.py \
  --config configs/decoder.yaml \
  --visualization videos/reachable_workspace.png
```

Or inspect it interactively with the MuJoCo mouse camera:

```bash
python scripts/visualize_reachable_workspace.py \
  --config configs/decoder.yaml \
  --live
```

### SAC training

A short integration run is useful before allocating a full budget:

```bash
python experiments/train_sac.py \
  --timesteps 10000 \
  --seed 0 \
  --run-dir outputs/sac/smoke
```

Run a larger experiment with the adaptive entropy default:

```bash
python experiments/train_sac.py \
  --timesteps 1000000 \
  --seed 0 \
  --ent-coef auto_0.3 \
  --eval-episodes 20 \
  --render-episodes 3 \
  --post-contact-seconds 4 \
  --run-dir outputs/sac/full_run
```

The command trains `MultiInputPolicy`, saves periodic evaluation data and the
best periodic model, saves the final checkpoint, evaluates deterministic
held-out seeds, and renders annotated episodes. Training displays an SB3
tqdm/rich progress bar by default; pass `--no-progress` for non-interactive log
files. The output layout is:

```text
outputs/sac/full_run/
├── final_model.zip
├── evaluation_summary.json
├── best/
│   └── best_model.zip
├── evaluations/
│   └── evaluations.npz
└── renders/
    ├── episode_00_seed_10000.mp4
    └── ...
```

The JSON summary records aggregate and per-episode metrics, evaluation seeds,
resolved task configuration, sampled episode parameters, checkpoint path, and
the configured/current entropy coefficient.

Evaluate and render the most recently modified checkpoint without training:

```bash
python experiments/train_sac.py \
  --eval-only \
  --run-dir outputs/sac/full_run \
  --eval-episodes 20 \
  --render-episodes 3
```

Choose a checkpoint explicitly when comparing final and best policies:

```bash
python experiments/train_sac.py \
  --eval-only \
  --run-dir outputs/sac/full_run \
  --checkpoint outputs/sac/full_run/best/best_model.zip
```

Eval-only results use `eval_only_summary.json` and `eval_only_renders/`, so they
do not overwrite the original post-training report.

### Weights & Biases

W&B tracking uses the `contact-deflection` project by default. Authenticate
once for online logging:

```bash
wandb login
```

Then add a project name to the normal training command:

```bash
python experiments/train_sac.py \
  --timesteps 1000000 \
  --seed 0 \
  --run-dir outputs/sac/full_run \
  --wandb-name sac-seed-0 \
  --wandb-tag full-run \
  --wandb-tag seed-0
```

This synchronizes SB3 TensorBoard metrics—including actor/critic losses,
entropy coefficient, episode return, and periodic deterministic evaluation—to
W&B. Final aggregate metrics, rendered videos, the evaluation summary, and the
policy checkpoint are also logged. Use `--wandb-project` or `WANDB_PROJECT` to
override the default project. `WANDB_ENTITY` and `WANDB_MODE` environment
variables are also supported. For a network-free run that can be synchronized
later, use:

```bash
python experiments/train_sac.py \
  --timesteps 1000000 \
  --run-dir outputs/sac/offline_run \
  --wandb-mode offline
```

Use `--wandb-mode disabled` to run without recording or uploading W&B data.

## Configuration

- `configs/env.yaml`: clocks, sensing, PD control, terminal reward, desired
  velocity distribution, and randomized episode parameters.
- `configs/decoder.yaml`: candidate workspace, IK, structured action scales,
  joint velocity limits, and trajectory derivative limits.
- `environment.yml`: reproducible Conda environment.

The current velocity, acceleration, jerk, and torque limits are intentionally
aggressive simulation settings and are not verified UR5 hardware ratings.

## Repository structure

```text
assets/mjcf/                         MuJoCo spacecraft/contact model
configs/                             Benchmark and decoder parameters
experiments/train_sac.py             Train/evaluate orchestration
scripts/visualize_reachable_workspace.py
src/contact_deflection/
├── envs/                            Gymnasium benchmark and simulator wrapper
├── estimation/                      Projectile belief and spacetime corridor
├── geometry/                        Contact-frame construction
├── kinematics/                      Workspace and collision-aware Mink IK
├── control/                         Decoder, twist fit, trajectory, torque control
├── rl/                              SAC training and deterministic evaluation
└── visualization/                   Offscreen and interactive visualization
tests/                               Fast deterministic scientific checks
third_party/                         Pinned upstream assets and license
scratch/                             Ignored one-off investigations and probes
```

## Known limitations

- Projectile motion is linear in the inertial world frame before contact; CW
  orbital acceleration is intentionally out of scope.
- Spacecraft pose and twist are treated as known by the estimator. Uncertain
  spacecraft localization would require an error-state EKF or joint filter.
- IK and terminal Jacobians freeze the current base pose. The planner does not
  predict generalized-Jacobian momentum coupling, although MuJoCo executes the
  resulting base reaction.
- Collision constraints apply to terminal IK iterations, not the full executed
  joint-space path. The quintic backend is not a collision-aware motion planner
  or minimum-jerk optimizer.
- Trajectory derivative violations are diagnostic; the current environment
  executes the reference whenever terminal IK is feasible.
- Action coordinates 6–7 are reserved for shield angular velocity but have zero
  scale in the current decoder configuration.
- The bus, solar arrays, shield, contact law, and aggressive actuator limits are
  simplified research models rather than flight-qualified hardware.
- The terminal-only reward and SAC hyperparameters require validation across
  longer runs and multiple seeds.

## Third-party provenance

The UR5 meshes and adapted arm model come from
[SpaceRobotEnv](https://github.com/Tsinghua-Space-Robot-Learning-Group/SpaceRobotEnv)
at the pinned commit recorded in `THIRD_PARTY.md`. The upstream Apache-2.0
license is reproduced under `third_party/space_robot_env/`.
