import neat
import random

class CustomIZGenome(neat.iznn.IZGenome):
    def __init__(self, key):
        super().__init__(key)
        self.simulation_steps_init_mean = 50
        self.simulation_steps_init_stdev = 20
        self.simulation_steps_min_value = 1
        self.simulation_steps_max_value = 100  
        self.simulation_steps_mutate_power = 0
        self.simulation_steps_mutate_rate = 0.5

        self.input_scaling_init_mean = 100.0
        self.input_scaling_init_stdev = 20.0
        self.input_scaling_min_value = 10.0
        self.input_scaling_max_value = 250.0
        self.input_scaling_mutate_power = 10.0
        self.input_scaling_mutate_rate = 0.5

        self.input_min_init_mean = -10.0
        self.input_min_init_stdev = 20.0
        self.input_min_min_value = -200.0
        self.input_min_max_value = 200.0
        self.input_min_mutate_power = 10.0
        self.input_min_mutate_rate = 0.5

        self.background_init_mean = 0.0
        self.background_init_stdev = 2.0
        self.background_min_value = -20.0
        self.background_max_value = 20.0
        self.background_mutate_power = 2.0
        self.background_mutate_rate = 0.5

        self.simulation_steps = int(self.simulation_steps_init_mean)
        self.input_scaling = self.input_scaling_init_mean
        self.input_min = self.input_min_init_mean
        self.background = self.background_init_mean

    def configure_new(self, config):
        super().configure_new(config)

        self.simulation_steps = int(random.gauss(
            self.simulation_steps_init_mean,
            self.simulation_steps_init_stdev
        ))
        self.simulation_steps = max(self.simulation_steps_min_value,
                                    min(self.simulation_steps_max_value, self.simulation_steps))

        self.input_scaling = random.gauss(
            self.input_scaling_init_mean,
            self.input_scaling_init_stdev
        )
        self.input_scaling = max(self.input_scaling_min_value,
                                 min(self.input_scaling_max_value, self.input_scaling))

        self.input_min = random.gauss(
            self.input_min_init_mean,
            self.input_min_init_stdev
        )
        self.input_min = max(self.input_min_min_value,
                             min(self.input_min_max_value, self.input_min))

        self.background = random.gauss(
            self.background_init_mean,
            self.background_init_stdev
        )
        self.background = max(self.background_min_value,
                              min(self.background_max_value, self.background))

    def mutate(self, config):
        super().mutate(config)

        if random.random() < self.simulation_steps_mutate_rate:
            delta = int(random.gauss(0, self.simulation_steps_mutate_power))
            self.simulation_steps = max(self.simulation_steps_min_value,
                                        min(self.simulation_steps_max_value,
                                            self.simulation_steps + delta))

        if random.random() < self.input_scaling_mutate_rate:
            delta = random.gauss(0, self.input_scaling_mutate_power)
            self.input_scaling = max(self.input_scaling_min_value,
                                     min(self.input_scaling_max_value,
                                         self.input_scaling + delta))

        if random.random() < self.input_min_mutate_rate:
            delta = random.gauss(0, self.input_min_mutate_power)
            self.input_min = max(self.input_min_min_value,
                                 min(self.input_min_max_value,
                                     self.input_min + delta))

        if random.random() < self.background_mutate_rate:
            delta = random.gauss(0, self.background_mutate_power)
            self.background = max(self.background_min_value,
                                  min(self.background_max_value,
                                      self.background + delta))

    def crossover(self, other, key, config):
        child = super().crossover(other, key, config)
        child.simulation_steps = random.choice([self.simulation_steps, other.simulation_steps])
        child.input_scaling = random.choice([self.input_scaling, other.input_scaling])
        child.input_min = random.choice([self.input_min, other.input_min])
        child.background = random.choice([self.background, other.background])
        return child
