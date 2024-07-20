import argparse
import time
from tqdm import tqdm
import numpy as np
from envs.episodic.navigate_T import NavigateT, EnvConfig


def main(width, height, random):
    config = EnvConfig(width=width, height=height)
    env = NavigateT(config=config)
    obs = env.reset()
    n_iter = 10000

    env.logger.info(f"{random=}, {width=}, {height=}")

    try:
        for _ in range(n_iter):
            if random:
                action = env.action_space.sample()
            else:
                action = 1
            obs, reward, done, truncated, info = env.step(action)
            env.render(mode="console")  # Use console mode for rendering

            if done or truncated:
                break

            # time.sleep(0.1)  # Add a small delay to make the rendering visible
    except KeyboardInterrupt:
        env.logger.info("Simulation interrupted by user")
    except Exception as e:
        env.logger.error(f"Exception: {e}")
        raise e
    finally:
        env.close()  # Ensure the environment is properly closed


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
    parser.add_argument(
        "--random",
        action="store_true",
        default=False,
        help="Whether to choose actions randomly else no control field is applied",
    )
    args = parser.parse_args()
    main(width=args.width, height=args.height, random=args.random)
