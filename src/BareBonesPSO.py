from Particle import Particle
import numpy as np


class BareBonesPSO:
    def __init__(self, bounds, particle_dim, particles_num):
        self.gbest_error = -1
        self.gbest = np.array([])

        self.swarm = []
        self.initialize_particles(particle_dim, particles_num, bounds)

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
                particle.update_position(self.gbest, func)
                pbest, pbest_error = particle.get_pbest()
                if self.gbest_error > pbest_error:
                    self.gbest_error = pbest_error
                    self.gbest = pbest

        return self.gbest, self.gbest_error

    def initialize_particles(self, particle_dim, particles_num, bounds):
        for i in range(particles_num):
            position = np.random.uniform(bounds[0], bounds[1], particle_dim)
            self.swarm.append(Particle(position))
