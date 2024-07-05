from .episodic.navigate_T import NavigateT
from .continuous.channel import Channel
from .continuous.channel_flow import ChannelFlow
from gymnasium.envs.registration import register


def make(env_name, **kwargs):
    envs = {
        "NavigateT-v0": NavigateT,
    }
    try:
        return envs[env_name](**kwargs)
    except KeyError:
        raise ValueError(f"Unknown environment name {env_name}")


register(
    id="NavigateT-v0",
    entry_point="envs.episodic.navigate_T:NavigateT",
)

register(
    id="Channel-v0",
    entry_point="envs.continuous.channel:Channel",
)

register(
    id="ChannelFlow-v0",
    entry_point="envs.continuous.channel_flow:ChannelFlow",
)

register(
    id="Rightward-v0",
    entry_point="envs.episodic.rightward:Rightward",
)
