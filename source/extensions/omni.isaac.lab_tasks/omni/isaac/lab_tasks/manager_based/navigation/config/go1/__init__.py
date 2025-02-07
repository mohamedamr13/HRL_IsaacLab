# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, go1_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Navigation-Flat-Go1-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go1_env_cfg.Go1NavigationEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.NavigationEnvPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Navigation-Flat-Go1-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go1_env_cfg.Go1NavigationEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.NavigationEnvPPORunnerCfg,
    },
)
