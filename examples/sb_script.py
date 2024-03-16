import os
from envs import make
from envs.episodic.navigate_T import NavigateT
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

import wandb
from wandb.integration.sb3 import WandbCallback

# Define your training parameters
config = {
    "algorithm": "PPO",
    "policy_type": "MlpPolicy",
    "n_total_timesteps": 10,
    "n_envs": 16,
    "gamma": 0.99,
}

# Initialize WandB
run = wandb.init(
    project="lvmc-channel-flow-v0",  # Make sure to replace with your actual project name
    config=config,
    sync_tensorboard=True,  # Auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # Auto-upload the videos of agents playing the game
    save_code=True,  # Optional: This will save your script in WandB for versioning
)

# Create the environment
env = make_vec_env(
    "ChannelFlow-v0",
    n_envs=config["n_envs"],
    env_kwargs={
        "width": 50,
        "height": 25,
        "density": 0.3,
        "control_interval": 1e-2,
        "g": 1.5,
        "v0": 100,
        "max_iterations": int(1e6),
    },
)

# Setup directories based on WandB run ID
experiment_path = os.path.join("./data", run.id)
os.makedirs(experiment_path, exist_ok=True)

logs_path = os.path.join(experiment_path, "logs")
os.makedirs(logs_path, exist_ok=True)

model_save_path = os.path.join(experiment_path, "models")
os.makedirs(model_save_path, exist_ok=True)

# Define the model with the WandB callback specified in the learn method
model = PPO(
    "MlpPolicy", env, verbose=1, gamma=config["gamma"], tensorboard_log=logs_path
)

# Start training
model.learn(
    total_timesteps=config["n_total_timesteps"],
    callback=WandbCallback(
        gradient_save_freq=1,
        model_save_path=model_save_path,
        verbose=2,
    ),
    progress_bar=True,
)

# After training, you might want to evaluate the model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
wandb.log({"mean_reward": mean_reward, "std_reward": std_reward})

# Close the WandB run
run.finish()

# Run the trained model for one episode and log the video
obs = env.reset()
for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render(
        mode="console"
    )  # this will render the environment in the console using rich library
    if done:
        break

    # We should save the output
