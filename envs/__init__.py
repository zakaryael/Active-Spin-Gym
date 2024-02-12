from .base_env import LVMCBaseEnv

from .episodic.navigate_T import NavigateT


def make(env_name, **kwargs):
    envs = {
        "NavigateT": NavigateT,
    }
    try:
        return envs[env_name](**kwargs)
    except KeyError:
        raise ValueError(f"Unknown environment name {env_name}")
