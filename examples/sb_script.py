import os
from envs import make
from envs.episodic.navigate_T import NavigateT, EnvConfig
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env

import wandb
from wandb.integration.sb3 import WandbCallback

import argparse
from dataclasses import asdict
import json

# wandb.require("core")


def main(args):
    config = {
        "algorithm": args.algorithm,
        "policy_type": args.policy_type,
        "n_total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "gamma": args.gamma,
        "env_config": {
            "width": args.width,
            "height": args.height,
            "control_interval": args.control_interval,
            "g": args.gravity,
            "v0": args.initial_velocity,
            "max_iterations": args.max_iterations,
        },
        "checkpoint_freq": args.checkpoint_freq,
    }

    # Initialize WandB
    run = wandb.init(
        project=args.project,
        config=config,
        sync_tensorboard=True,  # Auto-upload sb3's tensorboard metrics
        save_code=True,
    )
    # Make the environment configuration
    env_config = EnvConfig(
        width=args.width,
        height=args.height,
        control_interval=args.control_interval,
        g=args.gravity,
        v0=args.initial_velocity,
        max_iterations=args.max_iterations,
        env_log_dir=wandb.run.dir,
    )
    # Save the environment configuration
    with open("env_config.json", "w") as f:
        json.dump(asdict(env_config), f)
    # Create the environment
    env = make_vec_env(
        "NavigateT-v0", n_envs=config["n_envs"], env_kwargs={"config": env_config}
    )

    # Select the model based on the algorithm argument
    if config["algorithm"] == "PPO":
        model = PPO(
            config["policy_type"],
            env,
            verbose=1,
            gamma=config["gamma"],
            tensorboard_log=wandb.run.dir,
        )
    elif config["algorithm"] == "DQN":
        model = DQN(
            config["policy_type"],
            env,
            verbose=1,
            gamma=config["gamma"],
            tensorboard_log=wandb.run.dir,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {config['algorithm']}")

    # Start training with frequent checkpoints
    model.learn(
        total_timesteps=config["n_total_timesteps"],
        callback=WandbCallback(
            gradient_save_freq=config["checkpoint_freq"],
            model_save_freq=config["checkpoint_freq"],
            model_save_path=wandb.run.dir,  # Save models in the WandB run directory
            verbose=2,
        ),
        progress_bar=True,
    )

    # Close the WandB run
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a NavigateT environment with WandB."
    )
    parser.add_argument(
        "--algorithm", type=str, default="PPO", help="RL algorithm to use (PPO or DQN)"
    )
    parser.add_argument(
        "--policy_type",
        type=str,
        default="MlpPolicy",
        help="Policy type (e.g., MlpPolicy)",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=30_000_000,
        help="Total timesteps for training",
    )
    parser.add_argument("--n_envs", type=int, default=1, help="Number of environments")
    parser.add_argument("--gamma", type=float, default=1, help="Discount factor")
    parser.add_argument(
        "--width", type=int, default=32, help="Width of the environment"
    )
    parser.add_argument(
        "--height", type=int, default=24, help="Height of the environment"
    )
    parser.add_argument(
        "--control_interval", type=float, default=1e-3, help="Control interval"
    )
    parser.add_argument("--gravity", type=float, default=1.5, help="Gravity value")
    parser.add_argument(
        "--initial_velocity", type=float, default=100, help="Initial velocity"
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=int(1e4),
        help="Max iterations per episode",
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=10000,
        help="Checkpoint frequency for saving the model",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="lvmc-navigateT-v0.1.0",
        help="WandB project name",
    )

    args = parser.parse_args()
    main(args)
