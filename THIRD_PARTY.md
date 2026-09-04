# Third-Party Components

## SpaceRobotEnv UR5 assets

The UR5 mesh files in `third_party/space_robot_env/assets/stls/` and the source
model adapted as `assets/mjcf/space_robot_ur5.xml` come from:

- Project: SpaceRobotEnv
- Repository: https://github.com/Tsinghua-Space-Robot-Learning-Group/SpaceRobotEnv
- Commit: `155989c2ae94a3afeedf9b8601b6125d83b9c097`
- Upstream paths: `SpaceRobotEnv/assets/spacerobot/stls/` and
  `SpaceRobotEnv/assets/spacerobot/arm_v3.xml`
- License: Apache License 2.0; reproduced at
  `third_party/space_robot_env/LICENSE`

The adapted model adds the project shield, removes visualization-only frame
sites, and is embedded in this project's modern MuJoCo torque-control scene.
The structure of `envs/space_robot_env.py` preserves the upstream environment's
simple ownership of model, data, reset, stepping, and rendering while porting
it from legacy Gym/`mujoco_py` to Gymnasium and modern `mujoco`.
