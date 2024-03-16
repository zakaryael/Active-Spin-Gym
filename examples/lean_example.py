import argparse  # Import argparse for command-line parsing
from icecream import ic
from tqdm import tqdm
import numpy as np
from envs.continuous.channel import Channel
from envs.episodic.navigate_T import NavigateT
from envs.continuous.channel_flow import ChannelFlow


def main(width, height):
    env = ChannelFlow(
        width=width,
        height=height,
        density=0.3,
        control_interval=1e-3,
        g=1.5,
        v0=100,
        max_iterations=int(1e7),
    )
    _ = env.reset()

    n_iter = 1000000

    for i in range(n_iter):
        try:
            action = env.action_space.sample()
            obs, reward, done, _, info = env.step(1)

            env.render()

            if done or i == n_iter - 1:
                print(f"\ndone at step {i}, reward = {reward}, info = {info}")
                env.console.save_html("animation.html", clear=True)
                break
        except Exception as e:
            print("An error occurred:", e)
            raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run environment simulation with custom dimensions."
    )
    parser.add_argument(
        "--width", type=int, default=10, help="Width of the environment."
    )
    parser.add_argument(
        "--height", type=int, default=10, help="Height of the environment."
    )
    args = parser.parse_args()
    main(width=args.width, height=args.height)
