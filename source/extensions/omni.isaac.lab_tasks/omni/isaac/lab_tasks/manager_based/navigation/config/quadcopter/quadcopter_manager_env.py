
import math
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RandomizationTermCfg as RandTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.assets import Articulation, ArticulationCfg, AssetBaseCfg


import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.envs.ui import ManagerBasedRLEnvWindow
import omni.isaac.lab.sim as sim_utils
import torch



from omni.isaac.lab_assets import CRAZYFLIE_CFG  # isort: skip
from omni.isaac.lab.markers import CUBOID_MARKER_CFG  # isort: skip

drone_cfg_scene_entity = SceneEntityCfg("drone")


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = RandTerm(
        func=mdp.reset_root_joint_state,
        mode="reset",
        params={'asset_cfg': drone_cfg_scene_entity}
    )

    # init_desired_pos = RandTerm(
    #     func=mdp.init_desired_pos,
    #     mode="startup",
    # )


@configclass
class ActionsCfg:
    """Action terms for the MDP."""
    joint_efforts = mdp.JointTorqueActionCfg(
                    asset_name="drone",
                    scale=1.0, 
                    joint_names=[".*"],
                    )
    

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={'asset_cfg': drone_cfg_scene_entity})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={'asset_cfg': drone_cfg_scene_entity})
        projected_gravity = ObsTerm(func=mdp.projected_gravity, params={'asset_cfg': drone_cfg_scene_entity})
        desired_pos_base = ObsTerm(func=mdp.desired_pos_b, params={'asset_cfg': drone_cfg_scene_entity})

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardCfg:
    lin_vel = RewTerm(
        func = mdp.lin_vel_l2,
        weight= -0.05,
        params={'asset_cfg': drone_cfg_scene_entity}
    )
    ang_vel = RewTerm(
        func=mdp.ang_vel_l2,
        weight= -0.01,
        params={'asset_cfg': drone_cfg_scene_entity}
    )

    distance_to_goal = RewTerm(
        func=mdp.distance_to_goal_tanh,
        weight= 15.0,
        params={'asset_cfg': drone_cfg_scene_entity}
    )

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    pass

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.quadcopter_died, 
        params={'asset_cfg': drone_cfg_scene_entity}
    )


class QuadcopterEnvWindow(ManagerBasedRLEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env , window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadcopterScene(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # episode_length_s = 10.0
    # decimation = 2
    # num_actions = 4
    # num_observations = 12
    # num_states = 0
    # debug_vis = True
    # num_envs = 512
    #ui_window_class_type = QuadcopterEnvWindow
    # env_spacing = 2.5
    # simulation
    # sim: SimulationCfg = SimulationCfg(
    #     dt=1 / 100,
    #     render_interval=decimation,
    #     disable_contact_processing=True,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #         restitution=0.0,
    #     ),
    # )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
        mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        project_uvw=True,
        texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # scene
    #scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # robot
    drone: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Drone")
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

@configclass
class QuadcopterNavigationEnv(ManagerBasedRLEnvCfg):
    scene: QuadcopterScene = QuadcopterScene(num_envs=24, env_spacing=2.5)
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardCfg = RewardCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""
        self.desired_pos_w = torch.zeros(self.scene.num_envs, 3, device='cuda:0')
        # self.thrust = torch.zeros(self.scene.num_envs, 1, 3, device='cuda:0')
        # self.moment = torch.zeros(self.scene.num_envs, 1, 3, device='cuda:0')
        self.thrust_to_weight = 1.9
        self.moment_scale = 0.01
        #_robot = Articulation(self.scene.robot)
        #robot_mass = self.scene.robot.root_physx_view.get_masses()[0].sum()
        robot_mass = 0.027
        self.robot_mass = robot_mass
        gravity_magnitude = torch.tensor(self.sim.gravity, device='cuda:0').norm()
        self.gravity_magnitude = gravity_magnitude
        self.robot_weight = (robot_mass * gravity_magnitude).item()
        self.decimation = 2
        self.episode_length_s = 10.0
        #self._terrain = self.scene.terrain.class_type(self.scene.terrain)
            # simulation settings
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        self.max_episode_length = math.ceil(self.episode_length_s / (self.sim.dt * self.decimation))

                #self.ui_window_class_type: QuadcopterEnvWindow = QuadcopterEnvWindow(self)

        ## target pose 
        #self.desired_pos_w = torch.zeros(self.scene.num_envs, 3, device=self.sim.device)

        # # Logging
        # self.episode_sums = {
        #     key: torch.zeros(self.scene.num_envs, dtype=torch.float)
        #     for key in [
        #         "lin_vel",
        #         "ang_vel",
        #         "distance_to_goal",
        #     ]
        # }



        
    

        # self.episode_length_buf = torch.zeros(self.scene.num_envs, dtype=torch.long)
        #self.reset_terminated = torch.zeros(self.scene.num_envs, dtype=torch.bool)
        #self.reset_time_outs = torch.zeros_like(self.reset_terminated)
        #self.reset_buf = torch.zeros(self.scene.num_envs, dtype=torch.bool)
        #self.action = torch.zeros(self.scene.num_envs, self.scene.num_actions, device=self.si m.device)


class QuadcopterNavigationEnv_PLAY(QuadcopterNavigationEnv):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 24
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False