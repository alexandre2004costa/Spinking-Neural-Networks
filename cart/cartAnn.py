import numpy as np
import neat
import pygame
from cartPole import *
import multiprocessing

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitness = 0
    state = np.array([0, 0, 0.05, 0])
    
    while True:
        
        res = net.activate(state)
        action = 1 if res[0] > 0.5 else 0
        state = simulate_cartpole(action, state)
        x, _, theta, _ = state
        
        if abs(x) > position_limit or abs(theta) > angle_limit:
            break
            
        fitness += 1
        if fitness >= 100000:
            break
            
    return fitness

def visualize_genome(winner, config, generation):
    state = np.array([0, 0, 0.05, 0])
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    running = True
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        res = net.activate(state)
        action = 1 if res[0] > 0.5 else 0 # Sigmoid activation func is between 0 and 1
        state = simulate_cartpole(action, state)
        x, _, theta, _ = state

        message = ""
        if abs(x) > position_limit or abs(theta) > angle_limit:
            message = "Failed! Restarting..."
            state = np.array([0, 0, 0.05, 0])

        draw_cartpole(screen, state, generation, 0, 0, message)
        clock.tick(50)
        
    pygame.quit()

def run_neat(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                        neat.DefaultSpeciesSet, neat.DefaultStagnation, 
                        config_file)
    
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    generation = 0
    class GenerationReporter(neat.reporting.BaseReporter):
        def start_generation(self, gen):
            nonlocal generation
            generation = gen
            
    population.add_reporter(GenerationReporter())
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = population.run(pe.evaluate, 100)
    
    print('\nBest genome:\n', winner)
    visualize_genome(winner, config, generation)

if __name__ == '__main__':
    config_path = 'cart/cartAnn_config.txt'
    run_neat(config_path)