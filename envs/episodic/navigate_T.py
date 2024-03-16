# The task will be for a number of  particles to navigate a T shape and go upwards instead of downwards.

from envs.base_env import LVMCBaseEnv
from lvmc.core.particle_lattice import Orientation
import torch


class NavigateT(LVMCBaseEnv):
    """
    Environment for navigating a T shape.
    """

    def reset(self, seed=None):
        obs, _ = super().reset(seed)

        self.simulation.add_particle_flux(
            region=self.init_region,
            orientation=Orientation.RIGHT,
            n_particles=self.n_init_particles,
        )
        self.n_top = 0
        self.n_down = 0
        self.n_particles = self.simulation.lattice.n_particles

        obs = self._get_obs()
        return obs, {}

    def reward(self):

        if self.n_particles == self.simulation.lattice.n_particles:
            return -0.001

        self.n_particles = self.simulation.lattice.n_particles
        pos = self.simulation.lattice.id_to_position  # dictionary of particle positions

        # now compute the number of particles that have reached the top of the lattice
        n_top = 0
        n_down = 0
        for p in pos:

            if pos[p][1] == 0:
                n_top += 1
            elif pos[p][1] == self.height - 1:
                n_down += 1

        reward = (self.n_top - n_top) - (self.n_down - n_down)
        self.n_top = n_top
        self.n_down = n_down
        return reward / self.n_init_particles

    def is_done(self):
        return self.simulation.lattice.is_empty.item()

    def compute_lattice_topology(self) -> dict:
        return make_T_shape(self.width, self.height)


def make_T_shape(width, height):
    """
    Create a T shape for the environment.
    """
    cutoff = int(
        (height - 1) * 0.4
    )  # Use 30% of the lattice height as an upper/bottom wall.
    depth = int((width - 1) * 0.9)

    obstacles = torch.zeros((height, width), dtype=torch.bool)
    sinks = torch.zeros((height, width), dtype=torch.bool)
    sinks[0] = 1
    sinks[-1] = 1
    obstacles[:cutoff, 0:depth] = 1
    obstacles[-cutoff:, 0:depth] = 1
    obstacles[:, -1] = 1
    x1, x2, y1, y2 = 0, depth, cutoff, height - cutoff - 1
    init_region = [x1, int(x2 / 5), y1, y2]

    # compute the number of particles to be added as a function of the initial region size
    region_size = (init_region[1] - init_region[0] + 1) * (
        init_region[3] - init_region[2] + 1
    )
    n_init_particles = int(region_size * 0.9)
    return {
        "sinks": sinks,
        "obstacles": obstacles,
        "init_region": init_region,
        "n_init_particles": n_init_particles,
    }
