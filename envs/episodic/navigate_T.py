# The task will be for a number of  particles to navigate a T shape and go upwards instead of downwards.

from envs.base_env import LVMCBaseEnv, Orientation, EnvConfig, Optional

# from lvmc.core.particle_lattice import Orientation
import torch
import numpy as np


class NavigateT(LVMCBaseEnv):
    """
    Environment for navigating a T shape.
    """

    def _initialize_simulation(self, seed: Optional[int]) -> None:
        super()._initialize_simulation(seed=seed)
        obstacles, sinks, region, region_size = self.make_T_shape(self.config)
        self.simulation.add_obstacles(obstacles).add_sinks(sinks).add_particles(
            region=region, orientation=Orientation.RIGHT
        )
        self.simulation.build()
        self.n_particles = self.simulation.lattice.n_particles
        self.n_init = self.n_particles
        self.info["n_particles_survived"] = 0
        self.info["n_particles_lost"] = 0
        self.info.update(
            {
                "n_particles_escaped": 0,
                "exit_reward": 0,
                "completion_bonus": 0,
                "time_penalty": 0,
                "n_init": self.n_init,
            }
        )
        self.info
        self.logger.info("Environment initialized")

    def reward(self):
        pos = np.array(list(self.simulation.lattice.id_to_position.values()))
        n_survived = np.sum(pos[:, 1] == 0)
        n_lost = np.sum(pos[:, 1] == self.config.height - 1)
        n_escaped = np.sum(pos[:, 0] == 0)

        n_good_exit = n_survived - self.info["n_particles_survived"]
        n_bad_exit = (
            n_lost
            - self.info["n_particles_lost"]
            + (n_escaped - self.info["n_particles_escaped"])
        )

        self.info["n_particles_survived"] = n_survived
        self.info["n_particles_lost"] = n_lost
        self.info["n_particles_escaped"] = n_escaped

        # Calculate percentage of particles that have exited correctly

        correct_exit_percentage = self.info["n_particles_survived"] / self.n_init

        # Graduated completion bonus
        completion_bonus = 0
        if self.is_done():
            completion_bonus = correct_exit_percentage

        exit_reward = (n_good_exit - n_bad_exit) / self.n_init
        time_penalty = -1 / self.config.max_iterations

        w1 = 0.0
        w2 = 1.0
        w3 = 0.0

        reward = w1 * completion_bonus + w2 * exit_reward + w3 * time_penalty

        self.info["exit_reward"] += exit_reward
        self.info["completion_bonus"] += completion_bonus
        self.info["time_penalty"] += time_penalty

        return reward

    def is_done(self):
        return self.simulation.lattice.is_empty.item()

    @staticmethod
    def make_T_shape(config: EnvConfig):
        """
        Create a T shape for the environment.
        Args:
            config (EnvConfig): Configuration parameters for the environment.

        Returns:
            Tuple containing:
            - obstacles (torch.Tensor): Boolean tensor representing obstacles.
            - sinks (torch.Tensor): Boolean tensor representing sinks.
            - init_region (List[int]): Coordinates defining the initial region [x1, x2, y1, y2].
            - region_size (int): Size of the initial region.
        """
        cutoff = int(
            (config.height - 1) * config.t_shape_cutoff
        )  # Use 30% of the lattice height as an upper/bottom wall.
        depth = int((config.width - 1) * config.t_shape_depth)

        obstacles = torch.zeros((config.height, config.width), dtype=torch.bool)
        sinks = torch.zeros((config.height, config.width), dtype=torch.bool)
        sinks[0] = 1
        sinks[-1] = 1
        sinks[cutoff : config.height - cutoff, 0] = 1
        obstacles[:cutoff, 0:depth] = 1
        obstacles[-cutoff:, 0:depth] = 1
        obstacles[:, -1] = 1
        x1, x2, y1, y2 = 1, depth, cutoff, config.height - cutoff - 1
        init_region = [x1, x2, y1, y2]

        # compute the number of particles to be added as a function of the initial region size
        region_size = (init_region[1] - init_region[0] + 1) * (
            init_region[3] - init_region[2] + 1
        )

        return obstacles, sinks, init_region, region_size
