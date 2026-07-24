# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the differential inverse kinematics controller with the simulator.
"""

import argparse
import socket  # <-- UDP 통신을 위해 추가

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.assets import RigidObject, RigidObjectCfg

from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

from pathlib import Path
CURRENT_DIR = Path(__file__).parent.absolute()
USD_PATH = Path.home() / "workspace1"/ "robot_arm_usd" / "arm_gripper.usd"

# -------------------------------------------------------------
# UDP 통신 설정 (Teensy 보드 주소)
# -------------------------------------------------------------
TEENSY_IP = "192.168.1.15"
UDP_PORT = 5005
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


@configclass
class MyCustomSceneCfg(InteractiveSceneCfg):
    print("-" * 50)
    print(f"[DEBUG] USD Path: {USD_PATH.resolve()}")
    print(f"[DEBUG] Does File Exist: {USD_PATH.exists()}")
    print("-" * 50)
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
        prim_path = "{ENV_REGEX_NS}/Robot",
        spawn = sim_utils.UsdFileCfg(
            usd_path=str(USD_PATH),
        ),
        actuators={
                    "arm_joints": ImplicitActuatorCfg(
                        joint_names_expr=["shoulder_yaw","shoulder_pitch","shoulder_roll","elbow_pitch","lower_arm_roll","wrist_pitch"],
                        stiffness=800.0,
                        damping=40.0,
                    ),
                    "gripper_joints": ImplicitActuatorCfg(
                        joint_names_expr=["LeftGripperJoint"],
                        stiffness=10000.0,
                        damping=100.0,
                    )
                },
        init_state = ArticulationCfg.InitialStateCfg(
            pos = (0,0,0),
            rot=(0,0,0,0),
            joint_pos={
                "shoulder_yaw": 0.0,
                "shoulder_pitch": -1.0472,
                "shoulder_roll":0.0,
                "elbow_pitch": 2.0944,
                "lower_arm_roll": 0.0,
                "wrist_pitch": -1.0472,
                "LeftGripperJoint": 0.0,
            }
        )
    )

    cube = RigidObjectCfg(
        prim_path = "/World/Cube",
        spawn = sim_utils.CuboidCfg(
            size = (0.04, 0.04, 0.155),
            rigid_props = sim_utils.RigidBodyPropertiesCfg(),
            mass_props = sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props = sim_utils.CollisionPropertiesCfg(),
            visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color = (1.0,0.0,0.0), metallic = 0.2),
        ),
        init_state = RigidObjectCfg.InitialStateCfg(
            pos = (0.53,0.00911,0.0),
            rot = (1,0,0,0),
        ),
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    robot = scene["robot"]
    cube = scene["cube"]

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
        [0.1465, 0.1465, 0.2597, 0.9239, 0, 0, 0.3827],
    ]

    ee_goals = torch.tensor(ee_goals, device=sim.device)
    current_goal_idx = 0
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:] = ee_goals[current_goal_idx]

    robot_entity_cfg = SceneEntityCfg("robot", body_names=["gripper_base"])
    robot_entity_cfg.resolve(scene)

    ee_frame_idx = robot_entity_cfg.body_ids[0]

    all_joint_names = robot.data.joint_names

    print("-------------------------------")
    for idx, name in enumerate(all_joint_names):
        print(f"  Index [{idx}] : {name}")
    print("-------------------------------")

    joint_names = robot.data.joint_names
    left_finger_idx = joint_names.index("LeftGripperJoint")
    close_pose = torch.full((scene.num_envs, 1), -0.02, device=scene.device)
    open_pose = torch.full((scene.num_envs, 1), 0.0, device=scene.device)

    sim_dt = sim.get_physics_dt()
    count = 0

    arm_ik = 0
    action_reset = 0

    # 목표 각도 변수 초기화
    joint_pos_des = robot.data.default_joint_pos[:, 0:6].clone()

    # Simulation loop
    while simulation_app.is_running():
        # reset
        if count % 600 == 0:
            count = 0
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            current_goal_idx = 0

            action_reset = 1
            arm_ik = 1

            cube_state = cube.data.default_root_state.clone()
            cube_state[:,0] = 0.53
            cube_state[:,1] = 0.00911
            cube_state[:,2] = 0.0
            cube_state[:,7:13] = 0.0
            cube.write_root_state_to_sim(cube_state)

        elif count < 100:
            arm_ik = 1
            action_reset = 0

        elif count < 150:
            robot.set_joint_position_target(close_pose, joint_ids=[left_finger_idx])
            arm_ik = 0
            action_reset = 0

        elif count < 250:
            if count == 150:
                current_goal_idx = 1
                action_reset = 1
            else:
                action_reset = 0
            arm_ik = 1
            
        elif count <350:
            if count == 250:
                current_goal_idx = 2
                action_reset = 1
            else:
                action_reset = 0
            arm_ik = 1
            
        elif count <450:
            if count == 350:
                current_goal_idx = 3
                action_reset = 1
            else:
                action_reset = 0
            arm_ik = 1

        elif count <500:
            robot.set_joint_position_target(open_pose, joint_ids=[left_finger_idx])
            arm_ik = 0
            action_reset = 0

        elif count <600:
            if count == 500:
                current_goal_idx = 4
                action_reset = 1
            else:
                action_reset = 0
            arm_ik = 1

        if action_reset:
            ik_commands[:] = ee_goals[current_goal_idx]
            joint_pos_des = joint_pos[:, 0:6].clone()
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)

        if arm_ik:
            ee_jacobi_idx = ee_frame_idx - 1
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, [0,1,2,3,4,5]]
            ee_pose_w = robot.data.body_pose_w[:, ee_frame_idx]
            root_pose_w = robot.data.root_pose_w
            joint_pos = robot.data.joint_pos[:, 0:6]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        # 1. 시뮬레이션 내 가상 로봇 제어
        robot.set_joint_position_target(joint_pos_des, joint_ids=[0,1,2,3,4,5])

        # -------------------------------------------------------------
        # 2. UDP
        # -------------------------------------------------------------
        # Index 0 : shoulder_yaw (CAN ID 1)
        # Index 1 : shoulder_pitch (CAN ID 2)
        # Index 2 : shoulder_roll (CAN ID 3)
        # Index 3 : elbow_pitch (CAN ID 4)
        # Index 4 : lower_arm_roll (CAN ID 5)
        # Index 5 : wrist_pitch (CAN ID 6)
        # Index 6 : LeftGripperJoint (CAN ID 7)
        # Index 7 : RightGripperJoint (CAN ID 8)
        
        target_yaw = joint_pos_des[0, 0].item()
        target_pitch = joint_pos_des[0, 1].item()

        udp_msg = f"P,{target_yaw:.4f},{target_pitch:.4f}"
        udp_sock.sendto(udp_msg.encode(), (TEENSY_IP, UDP_PORT))
        # -------------------------------------------------------------

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