import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from lvmc.core.simulation import Simulation
from lvmc.core.particle_lattice import Orientation
from typing import Tuple, Optional
import sys
from rich.live import Live
from rich.console import Console
from rich.text import Text
from rich.layout import Layout
from rich.panel import Panel
from IPython.display import clear_output, display


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
        width: int,
        height: int,
        density: float,
        control_interval: float = 1e-3,
        g: float = 1.5,
        v0: float = 100,
        max_iterations: Optional[int] = int(1e5),
    ) -> None:
        """
        Initialize the LVMC base environment.

        :param g: Alignment sensitivity parameter.
        :param v0: Base transition rate for particle movement.
        :param width: Width of the particle lattice.
        :param height: Height of the particle lattice.
        :param density: Particle density in the lattice.
        :param control_interval: Time interval for control actions.
        """
        super(LVMCBaseEnv, self).__init__()

        self.g = g
        self.v0 = v0
        self.width = width
        self.height = height
        self.density = density
        self.control_interval = control_interval
        self.current_time = 0.0
        self.max_iterations = 10000
        self.current_iteration = 0

        # Define the lattice topology.
        self._set_topology()

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(4, self.height, self.width), dtype=np.uint8
        )
        self.cumulative_reward = 0.0
        self.max_iterations = max_iterations
        self._initialize_simulation()

        self.console = Console(record=True, force_terminal=True)
        self.live = None  # Placeholder for the Live instance

        print(
            f"Initialized LVMCBaseEnv with width={width}, height={height}, density={density}, control_interval={control_interval}, g={g}, v0={v0}, max_iterations={max_iterations}"
        )

    def _set_topology(self):
        """
        Set the topology of the environment.
        """
        topology = self.compute_lattice_topology()

        for key, value in topology.items():
            setattr(self, key, value)

    def _initialize_simulation(self) -> None:
        """
        Initialize the simulation object.
        """
        self.simulation = Simulation(
            width=self.width,
            height=self.height,
            density=self.density,
            g=self.g,
            v0=self.v0,
        )
        self.simulation.lattice.set_obstacles(self.obstacles)
        try:
            self.simulation.lattice.set_sinks(self.sinks)
            self.simulation.lattice.set_sources(self.sources)
        except AttributeError:
            pass

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, dict]:
        """
        Apply an action to the environment and update its state.

        :param action: The action to apply.
        :return: A tuple containing the new observation, reward, done flag, and additional info.
        """

        # set a clock. starts at 0.0 and increments by self.simulation.dt each step.
        # when the clock reaches self.control_interval, apply the action.

        clock = 0.0
        while clock < self.control_interval:
            if self.is_done():
                break
            self.simulation.run()
            clock += self.simulation.delta_t

        # apply the action
        self.simulation.apply_magnetic_field(action - 1)

        self.current_time = self.simulation.t
        self.current_iteration += 1

        reward = self.reward()
        self.cumulative_reward += reward
        done = self.is_done()
        self.info = {
            "time": self.current_time,
            "action": action,
            "density": self.simulation.lattice.density,
            "reward": reward,
            "cumulative_reward": self.cumulative_reward,
            "iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "n_particles": self.simulation.lattice.n_particles,
            "n_particles_right": self.simulation.lattice.particles[
                Orientation.RIGHT.value
            ]
            .sum()
            .item(),
            "n_particles_left": self.simulation.lattice.particles[
                Orientation.LEFT.value
            ]
            .sum()
            .item(),
        }
        truncated = self.current_iteration >= self.max_iterations
        return self._get_obs(), reward, done, truncated, self.info

    def reset(self, seed=None, options=None) -> torch.Tensor:
        """
        Reset the environment to an initial state.
        :param seed: The random seed to use for the environment.
        :param options: Additional options for the environment.
        :return: The initial observation of the environment.
        """
        if seed is not None:
            np.random.seed(seed)
        self._initialize_simulation()

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

        # Generate additional information as Rich Text. all the info is stored in self.info in different colors
        info_text = Text.assemble(
            Text(f"Time: {self.info['time']:.2f}", style="bold white"),
            Text(f" | "),
            Text(f"Action: {self.info['action']}", style="bold orange"),
            Text(f" | "),
            Text(f"Density: {self.info['density']:.2f}", style="bold pink"),
            Text(f" | "),
            Text(f"Iteration: {self.info['iteration']}", style="bold white"),
            Text(f" | "),
            Text(f"Reward: {self.info['reward']:.2f}", style="blue"),
            Text(f" | "),
            Text(
                f"Cumulative Reward: {self.info['cumulative_reward']:.2f}",
                style="bold blue",
            ),
            Text(f" | "),
            Text(
                f"Max Iterations: {self.info['max_iterations']}", style="bold magenta"
            ),
            Text(f" | "),
            Text(f"Particles: {self.info['n_particles']}", style="bold green"),
            Text(f" | "),
            Text(
                f"Particles Right: {self.info['n_particles_right']}", style="bold green"
            ),
            Text(f" | "),
            Text(
                f"Particles Left: {self.info['n_particles_left']}", style="bold green"
            ),
        )

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

    def compute_lattice_topology(self) -> dict:
        """
        Compute the lattice topology for the environment.

        :param width: The width of the lattice.
        :param height: The height of the lattice.
        :return: A dictionary containing the obstacles and sinks in the lattice.
        """
        pass
