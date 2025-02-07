# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RandomizationTermCfg as RandTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg, ArticulationCfg
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns, CameraCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
import omni.isaac.lab_tasks.manager_based.navigation.mdp as mdp
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.anymal_c.flat_env_cfg import AnymalCFlatEnvCfg
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.unitree_go1.flat_env_cfg import UnitreeGo1FlatEnvCfg
# source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/velocity/config/unitree_go1/flat_env_cfg.py
from functools import partial
from omni.isaac.lab.actuators import ImplicitActuatorCfg
# from omni.isaac.wheeled_robots.controllers import DifferentialController
# from omni.isaac.wheeled_robots.robots import WheeledRobot

from omni.isaac.lab_assets.unitree import UNITREE_GO1_CFG  # isort: skip
from omni.isaac.lab_assets.anymal import ANYMAL_C_CFG  # isort: skip
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab_assets import CRAZYFLIE_CFG  # isort: skip
from omni.isaac.lab.markers import CUBOID_MARKER_CFG  # isort: skip


go1_cfg = UnitreeGo1FlatEnvCfg()
#anymalc_config = AnymalCFlatEnvCfg()
turtlebot_config = ImplicitActuatorCfg(
            joint_names_expr=["wheel_left_joint", "wheel_right_joint"],
            effort_limit=1000.0,
            velocity_limit=10.2,
            stiffness=0.0,
            damping=3.0,
            )
TURTLEBOT3_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path='assets/turtlebot3_burger_0.usd',
        usd_path='omniverse://localhost/Users/roshdim1/assets/turtlebot3_burger_0.usd',
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
        rigid_body_enabled=True,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,  
            stabilization_threshold=0.001,
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={"wheel_left_joint": 0.0, "wheel_right_joint": 0.0},
    ),
    actuators = {
            "turtlebot": turtlebot_config
        }     
    )
import os 
print( 'cwd: ', os.getcwd() )
go1_cfg_scene_entity = SceneEntityCfg("Go1")
turtlebot_cfg_scene_entity = SceneEntityCfg("turtlebot")

@configclass
class EventCfg:
    """Configuration for events."""

    # reset_base_go1 = RandTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-1, 1), "y": (-1, 1), "yaw": (-3.14, 3.14)},
    #         "velocity_range": {
    #             "x": (-0.0, 0.0),
    #             "y": (-0.0, 0.0),
    #             "z": (-0.0, 0.0),
    #             "roll": (-0.0, 0.0),
    #             "pitch": (-0.0, 0.0),
    #             "yaw": (-0.0, 0.0),
    #         },
    #         "asset_cfg": go1_cfg_scene_entity
    #     },
    # )  
    reset_base = RandTerm(
        func=mdp.reset_root_joint_state,
        mode="reset",
        params={'asset_cfg': turtlebot_cfg_scene_entity}
    )




@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    # @configclass
    # class Go1PolicyCfg(ObsGroup):
    #     """Observations for policy group."""

    #     # observation terms (order preserved)
    #     base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg" : go1_cfg_scene_entity } )
    #     base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params= { "asset_cfg": go1_cfg_scene_entity} ) 
    #     projected_gravity = ObsTerm(func=mdp.projected_gravity, params={ "asset_cfg": go1_cfg_scene_entity } )
    #     # pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "go1_pose_command"})
    #     # height_scan = ObsTerm(
    #     #     func=mdp.height_scan,
    #     #     params={"sensor_cfg": SceneEntityCfg("height_scanner_go1")},
    #     #     noise=Unoise(n_min=-0.1, n_max=0.1),
    #     #     clip=(-1.0, 1.0),
    #     # )
    #     # camera_capture = ObsTerm(
    #     #     func=mdp.camera_capture,
    #     #     params={"sensor_cfg": SceneEntityCfg("camera")},
    #     # )
    #     velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "go1_base_velocity"})
    #     joint_pos = ObsTerm( func=mdp.joint_pos, params={ "asset_cfg": go1_cfg_scene_entity } , noise=Unoise(n_min=-0.01, n_max=0.01), clip=(-1.0, 1.0) )
    #     joint_vel = ObsTerm( func=mdp.joint_vel, params={ "asset_cfg": go1_cfg_scene_entity } , noise=Unoise(n_min=-1.5, n_max=1.5), clip=(-1.0, 1.0)   )
    #     actions = ObsTerm(  func =mdp.last_action, params={ "action_name": 'go1_pre_trained_policy_action'  } )  
    
    
    @configclass
    class MultiAgentPolicyCfg(ObsGroup):
        "Observations for policy group."

        # base_lin_vel_go1 = ObsTerm(func=mdp.base_lin_vel, params={ "asset_cfg": go1_cfg_scene_entity } )
        # projected_gravity_go1 = ObsTerm(func=mdp.projected_gravity, params={ "asset_cfg": go1_cfg_scene_entity } )
        # pose_command_go1 = ObsTerm(func=mdp.generated_commands, params={"command_name": "go1_pose_command"})
        # height_scan_go1 = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner_go1")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(-1.0, 1.0),
        # )

        base_lin_vel_turtlebot = ObsTerm(func=mdp.base_lin_vel, params={'asset_cfg': turtlebot_cfg_scene_entity})
        base_ang_vel_turtlebot = ObsTerm(func=mdp.base_ang_vel, params={'asset_cfg': turtlebot_cfg_scene_entity})
        # projected_gravity_turtlebot = Obs Term(func=mdp.projected_gravity, params={'asset_cfg': turtlebot_cfg_scene_entity})
        desired_pos_base_turtlebot = ObsTerm(func=mdp.desired_pos_b, params={'asset_cfg': turtlebot_cfg_scene_entity})





    # observation groups
    # go1_policy: Go1PolicyCfg = Go1PolicyCfg()
    policy: MultiAgentPolicyCfg = MultiAgentPolicyCfg()

multiagent_cfg = ObservationsCfg()


@configclass
class ActionsCfg:
    """Action terms for the MDP."""

    # go1_pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
    #     asset_name="Go1",
    #     # policy_path='omniverse://localhost/Users/roshdim1/assets/model_999.pt',
    #     policy_path='/home/user/roshdim1/.local/share/isaac-sim-4.0.0/IsaacLab/logs/rsl_rl/unitree_go1_flat/2024-07-10_13-52-32/exported/policy.pt',
    #     low_level_decimation=4,
    #     low_level_actions=mdp.JointPositionActionCfg(asset_name="Go1", joint_names=[".*"], scale=0.5, use_default_offset=True),
    #     low_level_observations=multiagent_cfg.go1_policy,
    # )

    joint_velocities = mdp.MobileJointVelocityActionCfg(
                    asset_name="turtlebot",
                    scale=10.0, 
                    joint_names=["wheel_left_joint", "wheel_right_joint"],
                    )
    
    #print( 'low level obs', LOW_LEVEL_ENV_CFG.observations.policy )



def generate_reward_terms(pose_command_name: str):
    termination_penalty = RewTerm(mdp.is_terminated, weight=-400.0)
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={"std": 2.0, "command_name": pose_command_name},
    )
    position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={"std": 0.2, "command_name": pose_command_name},
    )
    orientation_tracking = RewTerm(
        func=mdp.heading_command_error_abs,
        weight=-0.2,
        params={"command_name": pose_command_name},
    )
    return termination_penalty, position_tracking, position_tracking_fine_grained, orientation_tracking

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # termination_penalty, position_tracking_go1, position_tracking_fine_grained_go1, orientation_tracking_go1 = generate_reward_terms("go1_pose_command")
    
    lin_vel = RewTerm(
        func = mdp.lin_vel_l2,
        weight= 0.05,
        params={'asset_cfg': turtlebot_cfg_scene_entity}
    )
    ang_vel = RewTerm(
        func=mdp.ang_vel_l2,
        weight= 0.01,
        params={'asset_cfg': turtlebot_cfg_scene_entity}
    )

    distance_to_goal = RewTerm(
        func=mdp.distance_to_goal_tanh,
        weight= 15.0,
        params={'asset_cfg': turtlebot_cfg_scene_entity}
    )

@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    pass
    # go1_pose_command = mdp.UniformPose2dCommandCfg(
    #     asset_name="Go1",
    #     simple_heading=False,
    #     resampling_time_range=(8.0, 8.0),
    #     debug_vis=True,
    #     ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-3.0, 3.0), pos_y=(-3.0, 3.0), heading=(-math.pi, math.pi)),
    # )  


    # go1_base_velocity = mdp.UniformVelocityCommandCfg(
    #     asset_name="Go1",
    #     resampling_time_range=(10.0, 10.0),
    #     rel_standing_envs=0.02,
    #     rel_heading_envs=1.0,
    #     heading_command=True,
    #     heading_control_stiffness=0.5,
    #     debug_vis=False,
    #     ranges=mdp.UniformVelocityCommandCfg.Ranges(
    #         lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
    #     ),
    # )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # go1_base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("go1_contact_forces", body_names="trunk"), "threshold": 1.0},
    # )


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with multiple land robots."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    # Go1: ArticulationCfg = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Go1")
    turtlebot: ArticulationCfg = TURTLEBOT3_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Turtlebot")
    
    # sensors
    # height_scanner_go1 = RayCasterCfg(
    #         prim_path="{ENV_REGEX_NS}/Go1/trunk",
    #         offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #         attach_yaw_only=True,
    #         pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    #         debug_vis=True,
    #         mesh_prim_paths=["/World/ground"],
    #         )

    # go1_contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Go1/.*", history_length=3, track_air_time=True)

    # camera = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
    #     update_period=0.1,
    #     height=480,
    #     width=640,
    #     data_types=["rgb", "distance_to_image_plane"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    # )
    
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # envirnoment: AssetBaseCfg = AssetBaseCfg(
    #         prim_path = '/World/Environment',
    #         spawn = sim_utils.UsdFileCfg( usd_path = 'omniverse://localhost/Users/roshdim1/assets/test_env.usd' ) 
        
    #     )


@configclass
class MobileQuadrupedEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneEntityCfg = MySceneCfg()
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventCfg = EventCfg()

    curriculum: CurriculumCfg = CurriculumCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""
       
        #  self.scene.height_scanner_go1.prim_path = "{ENV_REGEX_NS}/Go1/trunk"
        # self.scene.height_scanner_anymalc.prim_path = "{ENV_REGEX_NS}/AnymalC/base"

        # go1_contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Go1/.*", history_length=3, track_air_time=True)
        # anymalc_contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/AnymalC/.*", history_length=3, track_air_time=True)

        # self.scene.go1_contact_forces = go1_contact_forces

        self.scene.num_envs = 256
        self.scene.env_spacing = 5.5

        # self.sim.dt = go1_cfg.sim.dt
        # self.sim.render_interval = go1_cfg.decimation
        # self.decimation = go1_cfg.decimation * 10
        # self.episode_length_s = self.commands.go1_pose_command.resampling_time_range[1]

        self.desired_pos_w = torch.zeros(self.scene.num_envs, 3, device='cuda:0')
        self.thrust_to_weight = 1.9
        self.moment_scale = 0.01
        robot_mass = 0.027
        self.robot_mass = robot_mass
        gravity_magnitude = torch.tensor(self.sim.gravity, device='cuda:0').norm()
        self.gravity_magnitude = gravity_magnitude
        self.robot_weight = (robot_mass * gravity_magnitude).item()
        self.decimation = 5
        self.episode_length_s = 10.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        self.max_episode_length = math.ceil(self.episode_length_s / (self.sim.dt * self.decimation))



        # if self.scene.height_scanner_go1 is not None:
        #     self.scene.height_scanner_go1.update_period = (
        #         self.actions.go1_pre_trained_policy_action.low_level_decimation * self.sim.dt
        #     )
        # if self.scene.height_scanner_anymalc is not None:
        #     self.scene.height_scanner_anymalc.update_period = (
        #         self.actions.anymalc_pre_trained_policy_action.low_level_decimation * self.sim.dt
        #     )

        # if self.scene.go1_contact_forces is not None:
        #     self.scene.go1_contact_forces.update_period = self.sim.dt

        # self.scene.camera = camera


class MobileQuadrupedEnvCfg_PLAY(MobileQuadrupedEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 64
        self.scene.env_spacing = 5.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
