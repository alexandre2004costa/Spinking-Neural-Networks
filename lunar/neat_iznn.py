import neat
import numpy as np
import time
from rate_iznn import RateIZNN 


def encode_input(state, min_vals, max_vals, I_min=20.0, I_max=100.0):
    norm_state = (state - min_vals) / (max_vals - min_vals)
    I_values = I_min + norm_state * (I_max - I_min)
    return I_values


def eval_genome(I_min, I_diff, background, simulateFunc, genomes, config):
    for _, genome in genomes:
        genome.fitness = simulateFunc(I_min, I_diff, background, genome, config)

def run(config_values, simulateFunc, config_file, guiFunc, num_Generations=50):

    config = neat.Config(neat.iznn.IZGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    def create_phenotype(genome):
        return RateIZNN.create(genome, config)
    
    neat.iznn.IZGenome.create_phenotype = create_phenotype

    config.genome_config.weight_init_mean = config_values["weight_init_mean"]
    config.genome_config.weight_init_stdev = config_values["weight_init_stdev"]
    config.genome_config.weight_max_value = config_values["weight_max_value"]
    config.genome_config.weight_min_value = config_values["weight_min_value"]
    config.genome_config.weight_mutate_power = config_values["weight_mutate_power"]
    config.genome_config.weight_mutate_rate = config_values["weight_mutate_rate"]
    config.genome_config.weight_replace_rate = config_values["weight_replace_rate"]

    pop = neat.population.Population(config)
    
    generation_reached = 0
    
    class GenerationReporter(neat.reporting.BaseReporter):
        def start_generation(self, generation):
            nonlocal generation_reached
            generation_reached = generation

    pop.add_reporter(GenerationReporter())
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)


    def eval_genome_wrapper(genomes, config):
        eval_genome(config_values["I_min"], config_values["I_diff"], config_values["background"], simulateFunc, genomes, config)
    
    winner = pop.run(eval_genome_wrapper, num_Generations)
    #print(winner)
    #guiFunc(winner, config, config_values["I_min"], config_values["I_diff"], config_values["background"], generation_reached)