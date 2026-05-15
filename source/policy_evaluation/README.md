# Policy Evaluation

This package evaluates pi0/openpi-style policies on data-collection task JSONs.
It is intentionally separate from `geniesim.benchmark` and does not start the
data-collection gRPC server, ROS2 publishers, or rosbag recording pipeline.

Example:

```bash
source source/policy_evaluation/assets.bashrc
python source/policy_evaluation/app.py \
  --config source/policy_evaluation/configs/galbot_pi0_eval.example.json
```

The top-level evaluation JSON is meta-task-like: `tasks[]` points to
data-collection single-task templates and provides per-task `episodes` and
`prompt`. `meta_task_file` can also point at an existing data-collection
meta-task JSON, while the evaluation JSON supplies policy/camera/run settings.

The first adapter is a generic 7DoF single-arm controller. The default pi0 state
is 8D: seven arm joints plus the raw gripper joint value. Policy actions may
contain more than 8 dimensions; dimensions `0:7` are interpreted as absolute arm
joint targets and dimension `7` controls the gripper (`1` opens, `0` closes).
Set `wait_for_gripper_close_hold` to make close actions block until the gripper
state machine reaches hold control before the next buffered policy action runs.
Gripper/object friction and hold parameters can be tuned with
`gripper_static_friction`, `gripper_dynamic_friction`,
`object_static_friction`, `object_dynamic_friction`, `gripper_close_max_force`,
`gripper_hold_stiffness`, and `gripper_hold_max_force`; leave friction values
as `null` to keep the task JSON and robot USD physics materials unchanged.
`target_object_mass_override` and `object_mass_overrides` are available for
mass experiments, and default to the task JSON mass values.
`enable_distractors` defaults to `false`; when disabled, only the task grasp
object and place target/container are loaded from the generated layout.
`success_check_interval` only observes whether the task success condition has
been met; it does not end the episode by itself.
Set `terminate_on_arm_reset` to stop long episodes once the controlled arm has
first moved away from the reset pose by `arm_reset_away_threshold` and later
returns within `arm_reset_tolerance` for `arm_reset_consecutive_steps` policy
steps.
Set `terminate_on_grasp_lost` to stop failed episodes after
`grasp_steps_threshold` policy steps when the arm TCP is farther than
`grasp_judge_distance_threshold` from the task grasp object. If task success has
already been observed, this grasp-lost check is ignored and the final checker
still decides success at the actual episode termination.
The example config uses the openpi Linden image keys `cam_head`, `cam_left`, and
`cam_right`; older servers can switch these through `adapter.image_keys`.

The third-person observer camera can be configured with a world-space
`position` and `target`; the GUI viewport is pointed at that same observer
camera when a viewport is available. The older `position` plus `quaternion`
format is still accepted for compatibility.

Action chunks can be inspected offline from an episode directory:

```bash
python source/policy_evaluation/scripts/plot_action_chunks.py \
  output/policy_evaluation/galbot_pi0_eval/<run_id>/episodes/<episode_dir>
```

The script writes `overview.png`, `checks.json`, and detailed sampled frames
under `sample_frames/` in the episode's `action_chunk_analysis/` directory. It
also accepts a whole run directory and will generate one analysis folder per
episode. The sampled frames use the saved MP4 videos to approximate the policy
input images because `policy_io.jsonl` stores image shapes, not raw image
arrays.
