# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Runs the same differential-IK simulation as multiple_ik_udp.py, sending the same
UDP commands to the Teensy board, while also plotting joint_pos_des for all 6 arm
joints in real time.
"""

import argparse
import socket  # <-- added for UDP communication

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Real-time plot of joint_pos_des from the differential IK controller.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--plot_every", type=int, default=5, help="Redraw the plot every N simulation steps.")
parser.add_argument("--window", type=float, default=10.0, help="Seconds of recent history to show (0 = full history).")
parser.add_argument(
    "--max_joint_vel",
    type=float,
    default=0.5,
    help="Max joint speed [rad/s]. Caps how fast joint_pos_des can change per step (0 = no limit).",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from collections import deque

import matplotlib.pyplot as plt
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

from pathlib import Path
CURRENT_DIR = Path(__file__).parent.absolute()
USD_PATH = Path.home() / "workspace1" / "robot_arm_usd" / "arm_gripper.usd"

JOINT_NAMES = ["shoulder_yaw", "shoulder_pitch", "shoulder_roll", "elbow_pitch", "lower_arm_roll", "wrist_pitch"]

# -------------------------------------------------------------
# UDP communication setup (Teensy board address)
# -------------------------------------------------------------
TEENSY_IP = "192.168.1.15"
UDP_PORT = 5005
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


@configclass
class MyCustomSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.01)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(USD_PATH),
        ),
        actuators={
                    "arm_joints": ImplicitActuatorCfg(
                        joint_names_expr=JOINT_NAMES,
                        stiffness=800.0,
                        damping=40.0,
                    ),
                    "gripper_joints": ImplicitActuatorCfg(
                        joint_names_expr=["LeftGripperJoint"],
                        stiffness=10000.0,
                        damping=100.0,
                    )
                },
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0),
            rot=(0, 0, 0, 0),
            joint_pos={
                "shoulder_yaw": 0.0,
                "shoulder_pitch": -1.0472,
                "shoulder_roll": 0.0,
                "elbow_pitch": 2.0944,
                "lower_arm_roll": 0.0,
                "wrist_pitch": -1.0472,
                "LeftGripperJoint": 0.0,
            }
        )
    )


class LiveJointPlot:
    """Rolling/real-time line plot of joint_pos_des, fed one sample at a time."""

    def __init__(self, joint_names: list[str], window: float, sim_dt: float):
        self.window = window
        maxlen = max(int(window / sim_dt), 1) if window > 0.0 else None
        self.time_hist: deque[float] = deque(maxlen=maxlen)
        self.joint_hist: list[deque[float]] = [deque(maxlen=maxlen) for _ in joint_names]

        plt.ion()
        self.fig, self.axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
        self.axes = self.axes.flatten()
        self.lines = []
        for ax, name in zip(self.axes, joint_names):
            (line,) = ax.plot([], [], lw=1.5)
            ax.set_title(name)
            ax.set_xlabel("time [s]")
            ax.set_ylabel("joint_pos_des [rad]")
            ax.grid(True, alpha=0.3)
            self.lines.append(line)
        self.fig.suptitle("joint_pos_des (real-time)")
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def append(self, t: float, joint_pos_des_row):
        self.time_hist.append(t)
        for j, val in enumerate(joint_pos_des_row):
            self.joint_hist[j].append(val)

    def redraw(self):
        if self.window > 0.0 and self.time_hist:
            t_min = self.time_hist[-1] - self.window
        else:
            t_min = None

        for ax, line, hist in zip(self.axes, self.lines, self.joint_hist):
            line.set_data(self.time_hist, hist)
            ax.relim()
            ax.autoscale_view()
            if t_min is not None:
                ax.set_xlim(left=max(t_min, 0.0), right=self.time_hist[-1] + 1e-6)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def closed(self) -> bool:
        return not plt.fignum_exists(self.fig.number)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    robot = scene["robot"]

    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    ee_goals = [
        [0.4, 0, 0.1, 1, 0, 0, 0],
        [0.3203, 0, 0.2597, 1, 0, 0, 0],
        [0.2265, 0.2265, 0.2597, 0.9239, 0, 0, 0.3827],
        [0.2265, 0.2265, 0.07, 0.9239, 0, 0, 0.3827],
    ]

    ee_goals = torch.tensor(ee_goals, device=sim.device)

    # Boomerang (back-and-forth) sequence: 0->1->2->3->4->3->2->1->(repeat)
    BOOMERANG_SEQUENCE = [0, 1, 2, 3, 2, 1]
    SEGMENT_STEPS = 250  # number of steps to hold each goal

    current_goal_idx = BOOMERANG_SEQUENCE[0]
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:] = ee_goals[current_goal_idx]

    robot_entity_cfg = SceneEntityCfg("robot", body_names=["gripper_base"])
    robot_entity_cfg.resolve(scene)

    ee_frame_idx = robot_entity_cfg.body_ids[0]

    sim_dt = sim.get_physics_dt()
    count = 0

    # Explicitly set this once at startup so the robot always starts from init_state(joint_pos).
    init_joint_pos = robot.data.default_joint_pos.clone()
    init_joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(init_joint_pos, init_joint_vel)
    robot.reset()

    # Initialize the target angle variable
    joint_pos_des = init_joint_pos[:, 0:6].clone()

    live_plot = LiveJointPlot(JOINT_NAMES, window=args_cli.window, sim_dt=sim_dt)

    # Simulation loop
    while simulation_app.is_running():
        if live_plot.closed():
            break

        seq_pos = count % (SEGMENT_STEPS * len(BOOMERANG_SEQUENCE))
        seq_idx = seq_pos // SEGMENT_STEPS
        action_reset = (seq_pos % SEGMENT_STEPS == 0)

        if action_reset:
            current_goal_idx = BOOMERANG_SEQUENCE[seq_idx]
            ik_commands[:] = ee_goals[current_goal_idx]
            if count > 0:
                joint_pos_des = robot.data.joint_pos[:, 0:6].clone()
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)

        ee_jacobi_idx = ee_frame_idx - 1
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, [0, 1, 2, 3, 4, 5]]
        ee_pose_w = robot.data.body_pose_w[:, ee_frame_idx]
        root_pose_w = robot.data.root_pose_w
        joint_pos = robot.data.joint_pos[:, 0:6]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        raw_joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        # Rate-limit the commanded joint step so motion speed is controlled directly,
        # instead of relying on SEGMENT_STEPS (which only sets dwell time at the goal;
        # the raw IK solve above jumps to the full target pose in ~1 step regardless).
        if args_cli.max_joint_vel > 0.0:
            max_delta = args_cli.max_joint_vel * sim_dt
            delta = torch.clamp(raw_joint_pos_des - joint_pos_des, -max_delta, max_delta)
            joint_pos_des = joint_pos_des + delta
        else:
            joint_pos_des = raw_joint_pos_des

        robot.set_joint_position_target(joint_pos_des, joint_ids=[0, 1, 2, 3, 4, 5])

        # -------------------------------------------------------------
        # UDP: CAN ID mapping and motor:link gear ratio conversion are handled entirely by the Teensy (main.cpp).
        # Here we simply send the simulation (link) frame angles (rate-limited joint_pos_des) as-is.
        # -------------------------------------------------------------
        target_yaw = joint_pos_des[0, 0].item()
        target_pitch = joint_pos_des[0, 1].item()
        target_roll = joint_pos_des[0, 2].item()
        target_elbow = joint_pos_des[0, 3].item()

        udp_msg = f"P,{target_yaw:.4f},{target_pitch:.4f},{target_roll:.4f},{target_elbow:.4f}"
        udp_sock.sendto(udp_msg.encode(), (TEENSY_IP, UDP_PORT))
        # -------------------------------------------------------------

        live_plot.append(count * sim_dt, joint_pos_des[0].detach().cpu().tolist())
        if count % args_cli.plot_every == 0:
            live_plot.redraw()

        scene.write_data_to_sim()

        sim.step()
        count += 1
        scene.update(sim_dt)

        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    scene_cfg = MyCustomSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
