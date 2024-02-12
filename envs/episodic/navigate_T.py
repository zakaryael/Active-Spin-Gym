# The task will be for a single particle to navigate a T shape and go upwards instead of downwards.

from envs.base_env import LVMCBaseEnv
import torch
from tqdm import tqdm


class NavigateT(LVMCBaseEnv):
    """
    Environment for navigating a T shape.
    """

    def __init__(self):
        super().__init__(
            width=40, height=10, density=0.0, control_interval=1e-3, g=1.0, v0=100
        )

        # compute a few parameters for the topology
        cutoff = int(
            (self.height - 1) * 0.4
        )  # Use 30% of the lattice height as an upper/bottom wall.
        depth = int((self.width - 1) * 0.9)

        # create a binary mask for the obstacles. 1 means there is an obstacle, 0 means there is not. a T shape is created.
        obstacles = torch.zeros((self.height, self.width), dtype=torch.bool)
        sinks = torch.zeros((self.height, self.width), dtype=torch.bool)
        sinks[0] = 1
        sinks[-1] = 1
        obstacles[:cutoff, 0:depth] = 1
        obstacles[-cutoff:, 0:depth] = 1
        obstacles[:, -1] = 1
        self.obstacles = obstacles
        self._initialize_simulation()
        self.simulation.add_particle(x=0, y=5)

    def reward(self):
        particles = self.simulation.lattice.particles
        # if there is a particle in the top row, give a reward of 1.0
        if particles[:, 0].any():
            return 1.0
        # if there is particle in the bottom row, give a reward of -1.0
        elif particles[:, -1].any():
            return -1.0
        else:
            return 0.0

    def is_done(self):
        particles = self.simulation.lattice.particles
        if particles[:, 0].any() or particles[:, -1].any():
            return True
        else:
            return False
