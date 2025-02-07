# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, multiagent_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Navigation-Flat-MultiAgent-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": multiagent_env_cfg.MutliAgentNavigationEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.NavigationEnvPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Navigation-Flat-MultiAgent-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": multiagent_env_cfg.MultiAgentNavigationEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.NavigationEnvPPORunnerCfg,
    },
)
