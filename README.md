# arm-ik — `udp` branch

Differential-IK control of a 6-DOF arm in Isaac Sim, with the solved joint targets
streamed to a Teensy board over UDP so the real arm follows the simulated one.

## Script flow

The three top-level scripts are not independent variants — each one is built by
extending the previous one:

```
multiple_ik.py  →  multiple_ik_udp.py  →  plot_joint_pos_des.py
```

- **[multiple_ik.py](multiple_ik.py)** — the baseline. Runs the differential IK
  controller against 5 end-effector goals on a fixed `count % 600` timer, and
  drives a gripper open/close + cube pick-and-place sequence. Sim-only, no UDP.
- **[multiple_ik_udp.py](multiple_ik_udp.py)** — same IK core, but the
  gripper/cube pick-and-place logic is stripped out. The goal schedule becomes a
  4-goal **boomerang sequence** (`0→1→2→3→2→1→…`, `SEGMENT_STEPS = 100`), and
  after every `compute()` call the resulting joint targets are packed into a
  string and sent over UDP to the Teensy.
- **[plot_joint_pos_des.py](plot_joint_pos_des.py)** — its own docstring says it
  directly: *"Runs the same differential-IK simulation as multiple_ik_udp.py,
  sending the same UDP commands... while also plotting joint_pos_des."* Same
  boomerang sequence (slower dwell: `SEGMENT_STEPS = 250`) and the same UDP
  send, plus two additions: a live matplotlib grid of all 6 joint targets, and
  a `--max_joint_vel` rate limiter (see below) applied to `joint_pos_des`
  before it's sent to the robot and over UDP.

So when changing goal logic or the UDP packet, `multiple_ik_udp.py` is the
source of truth that `plot_joint_pos_des.py` mirrors — check both when editing.

## `MyCustomSceneCfg`

Each script defines its own `MyCustomSceneCfg(InteractiveSceneCfg)`, an
Isaac Lab `@configclass` that declares everything to spawn in the scene:

- `ground` / `dome_light` — a ground plane and a dome light, standard scene dressing.
- `robot` — an `ArticulationCfg` that loads the arm from
  `~/workspace1/robot_arm_usd/arm_gripper.usd` and configures two actuator
  groups:
  - `arm_joints`: the 6 arm joints (`shoulder_yaw`, `shoulder_pitch`,
    `shoulder_roll`, `elbow_pitch`, `lower_arm_roll`, `wrist_pitch`), driven as
    implicit PD actuators (`stiffness=800`, `damping=40`).
  - `gripper_joints`: `LeftGripperJoint`, stiffer/harder-damped
    (`stiffness=10000`, `damping=100`) since it only needs to hold an
    open/closed position.

  `init_state.joint_pos` pins the arm's starting pose (elbow bent to
  `2.0944` rad, etc.) — this is what `robot.data.default_joint_pos` resolves
  to, and it's what the sim is explicitly reset to at the start of
  `run_simulator()`.
- `multiple_ik.py` additionally declares a `cube` (`RigidObjectCfg`) for the
  pick-and-place demo; the UDP scripts drop it since there's no gripper
  sequence anymore.

`InteractiveScene(scene_cfg)` instantiates this config into the actual USD
stage/physics scene; `scene["robot"]` is how the running script gets back a
handle to the spawned articulation.

## How the differential IK controller works

`DifferentialIKController` (`isaaclab.controllers`) implements the classic
Jacobian-based differential IK update:

```
Δq = J⁺ · Δx
q_desired = q_current + Δq
```

Configured here as `command_type="pose", ik_method="dls"` (damped least
squares / Levenberg–Marquardt), so instead of a plain pseudo-inverse it solves:

```
Δq = Jᵀ (J Jᵀ + λ²I)⁻¹ · Δx
```

The damping term `λ²I` keeps the solve well-conditioned near singularities
(where a plain pseudo-inverse would blow up), at the cost of some accuracy.

Each sim step, `run_simulator()`:
1. Reads the current EE pose in the robot's root frame
   (`subtract_frame_transforms`) and the geometric Jacobian for
   `gripper_base` from PhysX (`robot.root_physx_view.get_jacobians()`).
2. Calls `diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)`,
   which computes the full pose error (`Δx` = current EE pose vs. the goal set
   by `set_command()`) and returns `joint_pos + Δq` — i.e. **the entire joint
   target needed to close the pose error in one shot**, not a small
   per-step increment. `compute()` doesn't know about time or step size at all;
   it just resolves whatever pose error currently exists.
3. `robot.set_joint_position_target(joint_pos_des, ...)` hands that target to
   the PD actuators, which then drive the *simulated* joints toward it over
   many physics steps (limited by `stiffness`/`damping`, not by the IK call).

## Why `max_joint_vel` / delta-clamping was added

`diff_ik_controller.compute()` always solves for the full remaining pose
error, so the instant a new goal is set (`action_reset` / boomerang step),
`joint_pos_des` jumps to essentially the final target on the very next call —
regardless of `SEGMENT_STEPS`. `SEGMENT_STEPS` only controls how long a goal
is *held* before switching to the next one; it does nothing to pace the
transition itself.

In simulation that's masked because the PD actuators (`stiffness`/`damping`)
physically can't teleport the simulated joints even if the *target* jumps.
But the UDP-sent value is `joint_pos_des` itself, sent straight to the
Teensy/motors — so without rate-limiting, the real motors would receive a
target that jumps almost instantly to the new goal on every transition,
i.e. effectively commanding max velocity regardless of how "slow" the
boomerang schedule looks. That risks slamming the real hardware.

`plot_joint_pos_des.py` fixes this by clamping the per-step change in
`joint_pos_des` after the IK solve:

```python
max_delta = args_cli.max_joint_vel * sim_dt          # rad allowed per physics step
delta = torch.clamp(raw_joint_pos_des - joint_pos_des, -max_delta, max_delta)
joint_pos_des = joint_pos_des + delta
```

This turns the IK's one-shot jump into a bounded ramp: `joint_pos_des` moves
toward `raw_joint_pos_des` by at most `max_joint_vel` rad/s, every step,
independent of `SEGMENT_STEPS`. Both the simulated robot and the UDP packets
use this rate-limited value, so what's sent to the real hardware is bounded
by an explicit speed cap (`--max_joint_vel`, default `0.5` rad/s) instead of
by an implicit, goal-distance-dependent jump.

## UDP packet format

Both `multiple_ik_udp.py` and `plot_joint_pos_des.py` open a single UDP
socket at import time and send to the Teensy on every sim step:

```python
TEENSY_IP = "192.168.1.15"
UDP_PORT  = 5005
udp_msg = f"P,{target_yaw:.4f},{target_pitch:.4f},{target_roll:.4f},{target_elbow:.4f}"
```

- Comma-separated ASCII text, prefixed with a literal `P` (position command).
- 4 fields, each a 4-decimal-place float, in this fixed order:

  | field | joint           | HW CAN ID | note |
  |-------|-----------------|-----------|------|
  | 1     | `shoulder_yaw`   | 1         | motor:link gear ratio 4.8077:1 |
  | 2     | `shoulder_pitch` | 3         | motor:link gear ratio 3.180:1 |
  | 3     | `shoulder_roll`  | 2         | 1:1 |
  | 4     | `elbow_pitch`    | 4         | 1:1 |

- `lower_arm_roll`, `wrist_pitch`, and both gripper joints are **not** sent —
  those 4 hardware channels aren't wired up yet.
- The values sent are simulation (link-frame) joint angles in radians, taken
  straight from `joint_pos_des` (rate-limited, in `plot_joint_pos_des.py`).
  CAN ID mapping and the motor↔link gear-ratio conversion are done entirely
  on the Teensy (`main.cpp`) — this script never converts to motor-shaft
  units.

Example packet: `P,0.1234,-1.0472,0.0000,2.0944`

[test_udp.py](test_udp.py) is a standalone sender (2-value `P,<motor1>,<motor11>`
format, sine/cosine test signals) unrelated to the sim — useful for testing
the Teensy UDP link in isolation, without Isaac Sim running.
