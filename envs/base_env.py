import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from lvmc.core.simulation import Simulation
from lvmc.core.particle_lattice import Orientation
from typing import Tuple, Optional


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
        control_interval: float = 1e-4,
        g: float = 1.0,
        v0: float = 100,
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

        # Define the lattice topology.
        self.obstacles = torch.zeros((self.height, self.width), dtype=torch.bool)
        self.sinks = torch.zeros((self.height, self.width), dtype=torch.bool)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(4, self.height, self.width), dtype=np.bool_
        )

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
        self.simulation.lattice.set_sinks(self.sinks)

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
            self.simulation.run()
            clock += self.simulation.delta_t

        # apply the action
        self.simulation.apply_magnetic_field(action - 1)

        self.current_time = self.simulation.t

        reward = self.reward()
        done = self.is_done()
        info = {"time": self.current_time, "action": action}

        return self._get_obs(), reward, done, info

    def reset(self, seed=None) -> torch.Tensor:
        """
        Reset the environment to an initial state.

        :return: The initial observation of the environment.
        """
        if seed is not None:
            np.random.seed(seed)
        self._initialize_simulation()
        return self._get_obs()

    def render(self, mode: str = "console") -> None:
        """
        Render the environment to the screen or other mode.

        :param mode: The mode to render with.
        """
        if mode == "console":
            print(self.simulation.lattice)

    def close(self) -> None:
        """
        Perform any necessary cleanup.
        """
        pass

    def _get_obs(self) -> torch.Tensor:
        """
        Get the current observation from the simulation.

        :return: The current observation of the environment.
        """
        obs = self.simulation.lattice.query_lattice_state().numpy().astype(np.bool_)

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
