from envs.base_env import LVMCBaseEnv
from lvmc.core.particle_lattice import Orientation
from lvmc.core.simulation import Simulation
import torch


class ChannelFlow(LVMCBaseEnv):
    """
    Environment for navigating a channel with a Poiseuil flow.
    """

    def reset(self, seed=None):
        """
        Reset the environment to its initial state.
        """
        obs, _ = super().reset(seed)

        self.n_plus = 0
        self.n_minus = 0

        self._initialize_simulation()

        obs = self._get_obs()

        return obs, {}

    def reward(self):
        """
        Compute the reward for the current state.
        """
        # compute the number of particles that are oriented in the positive and negative directions
        n_plus = self.simulation.lattice.particles[Orientation.RIGHT.value].sum().item()
        n_minus = self.simulation.lattice.particles[Orientation.LEFT.value].sum().item()

        # compute the reward as the difference in the number of particles in the positive and negative directions
        reward = (self.n_plus - n_plus) - (self.n_minus - n_minus)
        self.n_plus = n_plus
        self.n_minus = n_minus
        return reward / self.simulation.lattice.n_particles

    def is_done(self):
        """
        Check if the environment is in a terminal state.
        """
        return self.current_iteration >= self.max_iterations - 1

    def compute_lattice_topology(self) -> dict:
        """
        Compute the lattice topology for the environment.
        """
        obstacles = torch.zeros((self.height, self.width), dtype=torch.bool)
        sinks = torch.zeros((self.height, self.width), dtype=torch.bool)
        sources = torch.zeros((self.height, self.width), dtype=torch.bool)
        obstacles[0, :] = True
        obstacles[-1, :] = True
        sources[:, 0] = True
        sinks[:, -1] = True
        return {"obstacles": obstacles, "sinks": sinks, "sources": sources}

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
            flow_params={"type": "poiseuille", "v1": 15},
        )
        self.simulation.lattice.set_obstacles(self.obstacles)
