import multiprocessing
import os
import random
import neat
import visualize
import numpy as np
from cartPole import *
import time
from multiprocessing import cpu_count



def encode_input(state, min_vals, max_vals, I_min=70.0, I_max=100.0):
    norm_state = (state - min_vals) / (max_vals - min_vals)
    I_values = I_min + norm_state * (I_max - I_min)
    return I_values

def simulate(I_min, I_diff, genome, config):
    #print(genome)
    net = neat.iznn.IZNN.create(genome, config)  

    state = np.array([0, 0, 0.05, 0])
    total_reward = 0
    
    steps_balanced = 0
    while True:
        input_values = encode_input(state, min_vals, max_vals, I_min, I_min + I_diff)
        net.set_inputs(input_values)
        output = net.advance(0.02)
        state = simulate_cartpole(int(output[0]), state)
        x, _, theta, _ = state
        if abs(x) > position_limit or abs(theta) > angle_limit:
            break
        
        total_reward += (math.cos(theta) + 1)
        steps_balanced += 1
        
        if steps_balanced >= 100000:  
            break

    
    return total_reward


def eval_genome(I_min, I_diff, genomes, config):
    for _, genome in genomes:
        genome.fitness = simulate(I_min, I_diff, genome, config)


param_ranges = {
    "pop_size": (100, 200),
    "bias_mutate_rate": (0.5, 1.0),
    "weight_mutate_rate": (0.5, 1.5),
    "conn_add_prob": (0.1, 0.5),
    "conn_delete_prob": (0.1, 0.5),
    "node_add_prob": (0.05, 0.3),
    "node_delete_prob": (0.05, 0.3),
    "compatibility_disjoint_coefficient": (0.5, 2.0),
    "compatibility_weight_coefficient": (0.2, 1.0),
    "max_stagnation": (20, 100),
    "I_min": (200.0, 600.0),
    "I_diff": (50, 500),
}

def random_config():
    return {param: random.uniform(*param_ranges[param]) if isinstance(param_ranges[param][0], float) 
            else random.randint(*param_ranges[param]) for param in param_ranges}

def run(config_values):

    config = neat.Config(neat.iznn.IZGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "oi.txt")
    
    config.pop_size = int(config_values["pop_size"])
    config.genome_config.bias_mutate_rate = config_values["bias_mutate_rate"]
    config.genome_config.weight_mutate_rate = config_values["weight_mutate_rate"]
    config.genome_config.conn_add_prob = config_values["conn_add_prob"]
    config.genome_config.conn_delete_prob = config_values["conn_delete_prob"]
    config.genome_config.node_add_prob = config_values["node_add_prob"]
    config.genome_config.node_delete_prob = config_values["node_delete_prob"]
    config.species_set_config.compatibility_disjoint_coefficient = config_values["compatibility_disjoint_coefficient"]
    config.species_set_config.compatibility_weight_coefficient = config_values["compatibility_weight_coefficient"]
    config.stagnation_config.max_stagnation = int(config_values["max_stagnation"])

    pop = neat.population.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)


    def eval_genome_wrapper(genomes, config):
        eval_genome(config_values["I_min"], config_values["I_diff"], genomes, config)
    
    winner = pop.run(eval_genome_wrapper, 10)
    '''
    print('\nBest genome:\n', winner)

    # Display the best genome
    state = np.array([0, 0, 0.05, 0])
    net = neat.iznn.IZNN.create(winner, config)  
    running = True
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        input_values = encode_input(state, min_vals, max_vals)
        net.set_inputs(input_values)
        output = net.advance(0.02)
        state = simulate_cartpole(int(output[0]), state)
        x, _, theta, _ = state
        message = ""
        if abs(x) > position_limit or abs(theta) > angle_limit:
            message = "Failed! Restarting..."
            net = neat.iznn.IZNN.create(winner, config)  
            state = np.array([0, 0, 0.05, 0])
            time.sleep(1)

        draw_cartpole(screen, state, 0, 0, 0, message)

        clock.tick(50)
    pygame.quit()'''
    return winner.fitness if winner else 0


def random_search(n_trials=10):
    best_config = None
    best_fitness = float('-inf')

    for _ in range(n_trials):
        config_values = random_config()
        fitness = run(config_values)
        if fitness > best_fitness:
            best_fitness = fitness
            best_config = config_values
            
        print(f"Test with config {config_values}, Fitness: {fitness}")
    
    print(f"Best configuration: {best_config}, Fitness: {best_fitness}")
    return best_config

if __name__ == '__main__':
    best_config = random_search(1000)