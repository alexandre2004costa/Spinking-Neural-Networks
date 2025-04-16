import neat
import numpy as np
import time

class BestGenomeReporter(neat.reporting.BaseReporter):
    def __init__(self):
        self.best_fitness = float('-inf')
        self.best_config = None
        self.generation = 0
    
    def post_evaluate(self, config, population, species, best_genome):
        self.generation += 1
        
        if best_genome.fitness > self.best_fitness:
            self.best_fitness = best_genome.fitness
            self.best_config = {
                'fitness': best_genome.fitness,
                'nodes': len(best_genome.nodes),
                'connections': len(best_genome.connections),
                'generation': self.generation
            }
            print("\nNew Best Configuration:")
            print(f"Generation: {self.generation}")
            print(f"Fitness: {self.best_fitness:.2f}")
            print(f"Nodes: {self.best_config['nodes']}")
            print(f"Connections: {self.best_config['connections']}")
            print("Network Parameters:")
            for node_id, node in best_genome.nodes.items():
                print(f"Node {node_id}: a={node.a:.3f}, b={node.b:.3f}, c={node.c:.3f}, d={node.d:.3f}")


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
    
    config.genome_config.weight_init_mean = config_values["weight_init_mean"]
    config.genome_config.weight_init_stdev = config_values["weight_init_stdev"]
    config.genome_config.weight_max_value = config_values["weight_max_value"]
    config.genome_config.weight_min_value = config_values["weight_min_value"]
    config.genome_config.weight_mutate_power = config_values["weight_mutate_power"]
    config.genome_config.weight_mutate_rate = config_values["weight_mutate_rate"]
    config.genome_config.weight_replace_rate = config_values["weight_replace_rate"]

    pop = neat.population.Population(config)
    
    generation_reached = 0
    
    best_reporter = BestGenomeReporter()
    pop.add_reporter(best_reporter)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)


    def eval_genome_wrapper(genomes, config):
        eval_genome(config_values["I_min"], config_values["I_diff"], config_values["background"], simulateFunc, genomes, config)
    
    winner = pop.run(eval_genome_wrapper, num_Generations)
    #print(winner)
    #guiFunc(winner, config, config_values["I_min"], config_values["I_diff"], config_values["background"], generation_reached)
