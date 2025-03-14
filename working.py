import numpy as np
import neat
import math
import pygame
from iznn import *
from cartPole import *


generation_number = 0
def eval_genomes(genomes, config):
    global generation_number
    generation_number += 1
    total_fitness = 0
    max_fitness = 0
    for genome_id, genome in genomes:   
        net = IZNN.create(genome, config)
        fitness = 0
        state = np.array([0, 0, 0.05, 0])
        for _ in range(int(max_time / time_step)):
            outputs = net.activate(state)  
            action = np.argmax(outputs)  
            state = simulate_cartpole(action, state)
            x, _, theta, _ = state
            if abs(x) > position_limit or abs(theta) > angle_limit:
                break
            fitness += (math.cos(theta) + 1)
        genome.fitness = fitness
        total_fitness += fitness
        max_fitness = max(max_fitness, fitness)
    avg_fitness = total_fitness / len(genomes)
    draw_cartpole(np.array([0, 0, 0.05, 0]), generation_number, avg_fitness, max_fitness)

def mean_aggregation(x):
    return sum(x) / len(x)

def run_neat(config_file):
    config = neat.Config(IZGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
    #config.genome_config.aggregation_function_defs.add('my_mean_function', mean_aggregation)

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(eval_genomes, 300)
    print('\nBest genome:\n', winner)

    # Display the best genome
    state = np.array([0, 0, 0.05, 0])
    net = IZNN.create(winner, config)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = np.argmax(net.activate(state))
        state = simulate_cartpole(action, state)
        x, _, theta, _ = state

        message = ""
        if abs(x) > position_limit or abs(theta) > angle_limit:
            message = "Failed! Restarting..."
            state = np.array([0, 0, 0.05, 0])

        draw_cartpole(state, generation_number, 0, 0, message)

        clock.tick(50)
    pygame.quit()

if __name__ == '__main__':
    config_path = 'config-feedforward.txt'
    run_neat(config_path)
