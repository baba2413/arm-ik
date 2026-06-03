# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""
This script demonstrates how to use the differential inverse kinematics controller with the simulator.


The differential IK controller can be configured in different modes. It uses the Jacobians computed by
PhysX. This helps perform parallelized computation of the inverse kinematics.


.. code-block:: bash


    # Usage
    ./isaaclab.sh -p scripts/tutorials/05_controllers/run_diff_ik.py


"""


"""Launch Isaac Sim Simulator first."""


import argparse


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

from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

from pathlib import Path
CURRENT_DIR = Path(__file__).parent.absolute()
USD_PATH = Path.home() / "workspace1"/ "robot_arm_usd" / "arm_gripper.usd"

@configclass
class MyCustomSceneCfg(InteractiveSceneCfg):
    print("-" * 50)
    print(f"[DEBUG] USD Path: {USD_PATH.resolve()}")
    print(f"[DEBUG] Does File Exist: {USD_PATH.exists()}")
    print("-" * 50)
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
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
                        joint_names_expr=[".*"],  # 모든 조인트에 적용 (필요시 정규식 수정 가능)
                        stiffness=800.0,          # 강성 (P Gain) - 로봇에 맞게 조절 필요
                        damping=40.0,             # 감쇠 (D Gain) - 로봇에 맞게 조절 필요
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
            }
        )
    )




def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]


    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)


    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))


    # Define goals for the arm
    # ee_goals = [
    #     [0.3, 0, 0.1, 0, 0, 0, 0],
    #     [0.3, 0, 0.3, 0, 0, 0, 0],
    #     [0.3, 0.1, 0.3, 0, 0, 0, 0],
    #     [0.3, 0.1, 0.1, 0, 0, 0, 0]
    # ]
    # ee_goals = [
    #     [0.3, 0, 0.1, 0, 0, 0, 0],
    #     [0.3, 0, 0.3, 0, 0, 0, 0],
    #     [0.2121, 0.2121, 0.3, 0.9239, 0, 0, 0.3827],
    #     [0.2121, 0.2121, 0.1, 0.9239, 0, 0, 0.3827]
    # ]
    ee_goals = [
        [0.4, 0, 0.1, 1, 0, 0, 0],
        [0.3203, 0, 0.2597, 1, 0, 0, 0],
        [0.2265, 0.2265, 0.2597, 0.9239, 0, 0, 0.3827],
        [0.2265, 0.2265, 0.1, 0.9239, 0, 0, 0.3827]
    ]

    ee_goals = torch.tensor(ee_goals, device=sim.device)
    # Track the given command
    current_goal_idx = 0
    # Create buffers to store actions
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
    close_pose = torch.full((scene.num_envs, 1), -0.027, device=scene.device)
    open_pose = torch.full((scene.num_envs, 1), 0.0, device=scene.device)



    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

    joint_pos = robot.data.default_joint_pos.clone()
    joint_pos_des = joint_pos[:, 0:6].clone()

    def reset_action():
        global joint_pos_des
        ik_commands[:] = ee_goals[current_goal_idx]
        joint_pos_des = joint_pos[:, 0:6].clone()
        # reset controller
        diff_ik_controller.reset()
        diff_ik_controller.set_command(ik_commands)

    def ik_compute():
        global joint_pos_des
        ee_jacobi_idx = ee_frame_idx - 1
        # obtain quantities from simulation
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, [0,1,2,3,4,5]]
        ee_pose_w = robot.data.body_pose_w[:, ee_frame_idx]
        root_pose_w = robot.data.root_pose_w
        joint_pos = robot.data.joint_pos[:, 0:6]
        # compute frame in root frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        # compute the joint commands
        joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

    # Simulation loop
    while simulation_app.is_running():
        # reset
        if count == 500:
            # reset time
            count = 0
            # reset joint state
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            # reset actions
            current_goal_idx = 0
            reset_action()

        elif count < 100:
            current_goal_idx = 0
            if count == 0: reset_action()
            ik_compute()
        elif count < 150:
            robot.set_joint_position_target(close_pose, joint_ids=[left_finger_idx])

        elif count < 250:
            current_goal_idx = 1
            if count == 150: reset_action()
            ik_compute()
            # robot.set_joint_position_target(close_pose, joint_ids=[left_finger_idx])
        elif count <350:
            current_goal_idx = 2
            if count == 250: reset_action()
            ik_compute()
            # robot.set_joint_position_target(close_pose, joint_ids=[left_finger_idx])
        elif count <450:
            current_goal_idx = 3
            if count == 350: reset_action()
            ik_compute()
        elif count <500:
            robot.set_joint_position_target(open_pose, joint_ids=[left_finger_idx])

        robot.set_joint_position_target(joint_pos_des, joint_ids=[0,1,2,3,4,5])

        scene.write_data_to_sim()


        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)


        # obtain quantities from simulation
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        # update marker positions
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])




def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = MyCustomSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)




if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

