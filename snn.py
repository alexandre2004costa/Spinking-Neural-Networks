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

    
    return steps_balanced


def eval_genome(I_min, I_diff, genomes, config):
    for _, genome in genomes:
        genome.fitness = simulate(I_min, I_diff, genome, config)

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
    
    winner = pop.run(eval_genome_wrapper, 100)
    
    print('\nBest genome:\n', winner)
    print('Best fitness:', winner.fitness)
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

        input_values = encode_input(state, min_vals, max_vals, config_values["I_min"], config_values["I_min"] + config_values["I_diff"])
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
    pygame.quit()
    return winner.fitness if winner else 0



if __name__ == '__main__':
   run({'pop_size': 156, 'bias_mutate_rate': 0.5529403627089193, 'weight_mutate_rate': 0.8940978000002908, 'conn_add_prob': 0.5, 'conn_delete_prob': 0.5, 'node_add_prob': 0.2, 'node_delete_prob': 0.2, 'compatibility_disjoint_coefficient': 0.8804122286986154, 'compatibility_weight_coefficient': 0.7509950987266565, 'max_stagnation': 39, 'I_min': -159.79031772078966, 'I_diff': 481})