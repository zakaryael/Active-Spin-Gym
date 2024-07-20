import stable_baselines3 as sb3
from envs.episodic.navigate_T import NavigateT, EnvConfig
import time
import argparse
import os
from stable_baselines3.common.evaluation import evaluate_policy


def main(
    model_path: str,
    g: float = 1.5,
    v0: float = 100,
    control_interval: float = 1e-3,
    max_iterations: int = 1e6,
    sleep: bool = True,
):
    # model path: strip the add additional path (/model) to the model path
    model_path = os.path.join(model_path, "model")

    # Load the trained model
    model = sb3.PPO.load(
        model_path
    )  # the model path should be of the form "data/<run_id>/models/model"
    _, height, width = model.observation_space.shape
    # Create the environment
    env_config = EnvConfig(
        width=width,
        height=height,
        control_interval=control_interval,
        g=g,
        v0=v0,
        max_iterations=max_iterations,
    )
    env = NavigateT(env_config)

    # Evaluate the model
    obs, _ = env.reset()
    for _ in range(max_iterations):
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)
        env.render()
        if sleep:
            time.sleep(0.1)

        if done:
            obs = env.reset()
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load a trained model and evaluate it in the NavigateT environment."
    )
    parser.add_argument(
        "--model_path", type=str, help="Path to the trained model file."
    )
    parser.add_argument(
        "--g", type=float, default=1.5, help="Gravitational acceleration."
    )
    parser.add_argument("--v0", type=float, default=10, help="Initial velocity.")
    parser.add_argument(
        "--control_interval", type=float, default=1e-3, help="Control interval."
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=int(1e6),
        help="Maximum number of iterations.",
    )
    parser.add_argument("--sleep", action="store_true", help="Sleep between steps.")

    args = parser.parse_args()
    main(
        args.model_path,
        args.g,
        args.v0,
        args.control_interval,
        args.max_iterations,
        args.sleep,
    )
