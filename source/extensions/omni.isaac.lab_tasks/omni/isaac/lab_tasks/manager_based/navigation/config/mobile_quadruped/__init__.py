# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Quacopter Navigation Manager environment.
"""

import gymnasium as gym

from . import agents
from .mobile_quadruped_env import MobileQuadrupedEnvCfg, MobileQuadrupedEnvCfg_PLAY

##
# Register Gym environments.
##

gym.register(
    id="Isaac-MobileQuadruped-Manager-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MobileQuadrupedEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.MobileQuadrupedPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-MobileQuadruped-Manager-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MobileQuadrupedEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.MobileQuadrupedPPORunnerCfg,
    },
)
