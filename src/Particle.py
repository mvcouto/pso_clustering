import numpy as np


class Particle:
    def __init__(self, x0):
        self.position = x0
        self.pbest = x0
        self.pbest_error = -1

    def update_position(self, gbest, func):
        mean = (self.pbest+gbest)/2
        std = np.abs(self.pbest-gbest)

        self.position = np.random.normal(mean, std, len(mean))

    def update_pbest(self, func):
        error = func(self.position)
        if error < self.pbest_error or self.pbest_error < 0:
            self.pbest = self.position
            self.pbest_error = error

    def get_pbest(self):
        return self.pbest, self.pbest_error
