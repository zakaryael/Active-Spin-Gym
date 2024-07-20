from lvmc.core.simulation import Simulation
from lvmc.core.particle_lattice import Orientation

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from typing import Tuple, Optional
import sys
from rich.live import Live
from rich.console import Console
from rich.text import Text
from rich.layout import Layout
from rich.panel import Panel
from IPython.display import clear_output, display

import logging
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
import os


@dataclass
class EnvConfig:
    width: int
    height: int
    control_interval: float = 1e-2
    g: float = 1.5
    v0: float = 100
    max_iterations: int = int(1e5)
    t_shape_cutoff: float = 0.4
    t_shape_depth: float = 0.8
    env_log_dir: Optional[str] = None


class LVMCBaseEnv(gym.Env):
    """
    Base environment for Lattice Vicsek Model with Magnetic Control (LVMC).

    This environment interfaces with the LVMC package to provide a Gym-compatible
    environment for controlling particle dynamics on a lattice using a magnetic field.

    Attributes:
        g (float): Alignment sensitivity parameter.
        v0 (float): Base transition rate for particle movement.
        width (int): Width of the particle lattice.
        height (int): Height of the particle lattice.
        density (float): Particle density in the lattice.
        control_interval (float): Time interval for control actions.
    """

    def __init__(
        self,
        config: EnvConfig,
    ) -> None:
        """
        Initialize the LVMC base environment.

        :param g: Alignment sensitivity parameter.
        :param v0: Base transition rate for particle movement.
        :param width: Width of the particle lattice.
        :param height: Height of the particle lattice.
        :param control_interval: Time interval for control actions.
        """
        super(LVMCBaseEnv, self).__init__()

        self.config = config
        self._validate_config()
        self.logger = logging.getLogger(self.__class__.__name__)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.MultiBinary(
            (4, self.config.height, self.config.width)
        )
        # self.observation_space = spaces.Box(
        #     low=0, high=1, shape=(4, self.height, self.width), dtype=np.uint8
        # )
        self.cumulative_reward = 0.0

        self.current_iteration = 0
        self.info = {
            "n_init": 0,
            "n_particles": 0,
            "max_iterations": self.config.max_iterations,
            "iteration": 0,
            "time": 0.0,
            "action": 0,
            "reward": 0.0,
        }
        self._initialize_simulation(seed=None)

        self.console = Console(record=True, force_terminal=True)
        self.live = None  # Placeholder for the Live instance
        self._setup_logger()

    def _setup_logger(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        # Check if a log_dir is specified in the config
        if hasattr(self.config, "env_log_dir") and self.config.env_log_dir is not None:
            log_dir = self.config.env_log_dir
        else:
            # Create logs directory if it doesn't exist
            if not os.path.exists("logs"):
                os.makedirs("logs")
            log_dir = "logs"

        # File Handler
        file_handler = RotatingFileHandler(
            f"{log_dir}/{self.__class__.__name__}_env.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=100,
        )
        print(f"Logging to {log_dir}/{self.__class__.__name__}_env.log")
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)

        #  Add both handler to the logger
        self.logger.addHandler(file_handler)

    def _validate_config(self) -> None:
        """
        Validate the configuration parameters.
        """
        if self.config.width <= 0 or self.config.height <= 0:
            raise ValueError("Width and height must be positive integers.")

    def _initialize_simulation(self, seed: Optional[int]) -> None:
        """
        Initialize the simulation object.
        """
        self.simulation = (
            Simulation(self.config.g, self.config.v0, seed=seed)
            .add_lattice(width=self.config.width, height=self.config.height)
            .add_control_field()
        )

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, bool, dict]:
        """
        Apply an action to the environment and update its state.

        :param action: The action to apply.
        :return: A tuple containing the new observation, reward, done flag, and additional info.
        """

        try:
            while self.simulation.t < (
                self.config.control_interval * (self.current_iteration + 1)
            ):
                if self.is_done():
                    break
                self.simulation.run()
            # apply the action
            self.simulation.apply_magnetic_field(action - 1)

            self.current_time = self.simulation.t
            self.current_iteration += 1

            reward = self.reward()
            self.cumulative_reward += reward
            self.info.update(
                {
                    "time": self.current_time,
                    "action": action,
                    "density": self.simulation.lattice.density,
                    "reward": reward,
                    "cumulative_reward": self.cumulative_reward,
                    "iteration": self.current_iteration,
                    "max_iterations": self.config.max_iterations,
                    "n_particles": self.simulation.lattice.n_particles,
                }
            )
            # update the logger with info
            done = self.is_done()
            truncated = self.current_iteration >= self.config.max_iterations
            if done or truncated:
                if done:
                    self.logger.info("Episode finished")
                if truncated:
                    self.logger.info("Episode truncated")

                # log nicely formated info from the self.info dict, one item per line
                self.logger.info("Episode info:")
                for key, value in self.info.items():
                    self.logger.info(f"{key}: {value}")

            return self._get_obs(), reward, done, truncated, self.info
        except Exception as e:
            self.logger.error(f"Error in step: {e}")
            sys.exit(1)

    def reset(self, seed=None, options=None) -> torch.Tensor:
        """
        Reset the environment to an initial state.
        :param seed: The random seed to use for the environment.
        :param options: Additional options for the environment.
        :return: The initial observation of the environment.
        """
        self._initialize_simulation(seed=seed)
        self.current_iteration = 0

        obs = self._get_obs()
        return obs, {}

    def render(self, mode="console"):
        if mode == "console":
            if self.live is None:
                # Initialize the Live object if it hasn't been already
                self.live = Live(console=self.console, auto_refresh=True)
                self.live.start()

            # Generate the content to be displayed
            content = self._generate_render_content()
            # Update the Live display
            self.live.update(content)
        if mode == "notebook":
            # Generate the content to be displayed
            content = self._generate_render_content()
            # Display the content in the notebook
            display(content)
            clear_output(wait=True)

    def _generate_render_content(self):
        # Generate the lattice visualization, assuming it returns Rich-formatted text
        lattice_visualization = self.simulation.lattice.visualize_lattice()
        lattice_text = Text.from_markup(lattice_visualization)

        # Known keys with specific styles
        known_styles = {
            "time": "bold white",
            "action": "bold orange",
            "iteration": "bold white",
            "reward": "blue",
            "cumulative_reward": "bold blue",
            "max_iterations": "bold magenta",
        }

        # Generate additional information as Rich Text dynamically
        info_text_parts = []
        for key, value in self.info.items():
            # Determine style
            if key in known_styles:
                style = known_styles[key]
            elif "survived" in key.lower():
                style = "bold green"
            elif "lost" in key.lower():
                style = "bold red"
            else:
                style = ""

            # Add Text object to list
            info_text_parts.append(
                Text(
                    f"{key.replace('_', ' ').title()}: {f"{value:.3f}".rstrip('0').rstrip('.')}",
                    style=style,
                )
            )
            info_text_parts.append(Text(" | "))

        # Remove the last separator
        if info_text_parts:
            info_text_parts.pop()

        info_text = Text.assemble(*info_text_parts)

        # Create the layout and add the lattice visualization and info text to it
        layout = Layout()
        layout.split_column(
            Layout(name="main", ratio=3),  # The main lattice visualization
            Layout(name="footer"),  # The footer with statistics
        )

        # Create panels for both sections
        lattice_panel = Panel(lattice_text, title="Lattice Visualization")
        info_panel = Panel(info_text, title="Statistics")

        # Update panels with content
        layout["main"].update(lattice_panel)
        layout["footer"].update(info_panel)

        return layout

    def close(self) -> None:
        """
        Perform any necessary cleanup.
        """
        if self.live is not None:
            self.live.stop()  # Stop the live display

    def _get_obs(self) -> torch.Tensor:
        """
        Get the current observation from the simulation.

        :return: The current observation of the environment.
        """
        obs = self.simulation.lattice.query_lattice_state().numpy().astype(np.uint8)

        return obs

    def reward(self) -> float:
        """
        Calculate the reward for the current state of the environment.

        :return: The reward for the current state of the environment.
        """
        pass

    def is_done(self) -> bool:
        """
        Check if the environment is in a terminal state.

        :return: A boolean indicating if the environment is in a terminal state.
        """
        pass
