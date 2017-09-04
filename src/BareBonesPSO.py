from Particle import Particle
import numpy as np


class BareBonesPSO:
    def __init__(self, bounds, particle_dim, particles_num):
        self.bounds = bounds

        self.gbest_error = -1
        self.gbest = np.array([])

        self.swarm = []
        self.initialize_particles(particle_dim, particles_num)

    def initialize_gbest(self, func):
        for particle in self.swarm:
            particle.update_pbest(func)
            pbest, pbest_error = particle.get_pbest()
            if self.gbest_error < 0 or self.gbest_error > pbest_error:
                self.gbest_error = pbest_error
                self.gbest = pbest

    def optimize(self, niter, func):
        self.initialize_gbest(func)

        for it in range(0, niter):
            for particle in self.swarm:
                particle.update_position(self.gbest, self.bounds)
                particle.update_pbest(func)
                pbest, pbest_error = particle.get_pbest()
                if self.gbest_error > pbest_error:
                    self.gbest_error = pbest_error
                    self.gbest = pbest

        return self.gbest, self.gbest_error

    def initialize_particles(self, particle_dim, particles_num):
        for i in range(0, particles_num):
            position = []
            for j in range(0, particle_dim):
                position.append(np.random.uniform(self.bounds[j][0], self.bounds[j][1], 1))
            self.swarm.append(Particle(np.concatenate(position)))
