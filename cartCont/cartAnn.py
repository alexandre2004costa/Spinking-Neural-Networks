import numpy as np
import neat
import pygame
from cartPole import *
import multiprocessing
import time


def decode_output(output):
     # Supondo que output[0] ∈ [-1, 1] (tanh)
    action = np.clip(output[0], -1.0, 1.0) * 10.0
    return action

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitness = 0
    state = np.array([0, 0, 0.05, 0])
    
    while True:
        input_values = np.array([state[0], state[2]]) # Taking out velocitys
        output = net.activate(input_values)
        action = decode_output(output)
        state = simulate_cartpole(action, state)
        x, _, theta, _ = state
        
        if abs(x) > position_limit or abs(theta) > angle_limit:
            break
            
        fitness += 1
        if fitness >= 5000:
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
                
        input_values = np.array([state[0], state[2]]) # Taking out velocitys
        output = net.activate(input_values)
        action = decode_output(output)
        state = simulate_cartpole(action, state)
        x, _, theta, _ = state
        
        if abs(x) > position_limit or abs(theta) > angle_limit:
            break
            
        message = ""
        if abs(x) > position_limit or abs(theta) > angle_limit:
            message = "Failed! Restarting..."
            state = np.array([0, 0, 0.05, 0])

        draw_cartpole(screen, state, generation, 0, 0, message)
        clock.tick(50)
        
    pygame.quit()

def run_neat(config_file):
    start_time = time.time()
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
    winner = population.run(pe.evaluate, 2000)
    
    elapsed_time = time.time() - start_time
    print('\nBest genome:\n', winner)
    print(f"Total time : {elapsed_time:.2f} sec")
    visualize_genome(winner, config, generation)

if __name__ == '__main__':
    config_path = 'cartCont/cartAnn_config.txt'
    run_neat(config_path)