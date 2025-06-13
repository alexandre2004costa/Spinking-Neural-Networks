import numpy as np
import neat
import pygame
import multiprocessing
import time
from stats import RLStatsCollector
from cart.cartPole import *

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

def run(config_file, num_Generations=50):  
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
    winner = population.run(pe.evaluate, num_Generations)
    
    elapsed_time = time.time() - start_time
    print('\nBest genome:\n', winner)
    print(f"Total time : {elapsed_time:.2f} sec")
    visualize_genome(winner, config, generation)


def run_experiment(config_file, num_Generations=50):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                        neat.DefaultSpeciesSet, neat.DefaultStagnation, 
                        config_file)
    fitness_threshold = config.fitness_threshold if hasattr(config, "fitness_threshold") else 100000
    stats_collector = RLStatsCollector(fitness_threshold=fitness_threshold)
    stats_collector.start_experiment()

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(False))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    generation = 0
    class GenerationReporter(neat.reporting.BaseReporter):
        def start_generation(self, gen):
            nonlocal generation
            generation = gen
            self.generation_start_time = time.time()
        def post_evaluate(self, config, population, species, best_genome):
            gen_time = time.time() - self.generation_start_time
            stats_collector.record_generation(best_genome.fitness, gen_time)

    population.add_reporter(GenerationReporter())
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = population.run(pe.evaluate, num_Generations)

    stats_collector.end_experiment()
    elapsed_time = stats_collector.learning_time()

    # Evaluate winner average time to take an action
    state = np.array([0, 0, 0.05, 0])
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    for _ in range(1000):
        t0 = time.time()
        res = net.activate(state)
        action = 1 if res[0] > 0.5 else 0
        t1 = time.time()
        stats_collector.record_action_time(t1 - t0)
        state = simulate_cartpole(action, state)
        x, _, theta, _ = state
        if abs(x) > position_limit or abs(theta) > angle_limit:
            break

    best_fitness = max(stats_collector.fitness_history) if stats_collector.fitness_history else 0
    mean_fitness = np.mean(stats_collector.fitness_history) if stats_collector.fitness_history else 0
    success_generation = stats_collector.success_generation if stats_collector.success_generation else num_Generations
    success = 1 if stats_collector.success_generation else 0
    mean_decision_time = stats_collector.mean_decision_time() if stats_collector.action_times else 0
    num_hidden = config.genome_config.num_hidden if hasattr(config.genome_config, "num_hidden") else None

    return {
        "learning_time": elapsed_time,
        "success": success,
        "success_generation": success_generation,
        "mean_decision_time": mean_decision_time,
        "best_fitness": best_fitness,
        "mean_fitness": mean_fitness,
        "total_generations": generation+1,
        "hidden_nodes": num_hidden
    }

if __name__ == '__main__':
    print("running Ann experiment")
    run('cartCont/cartAnn_config.txt', 1000)