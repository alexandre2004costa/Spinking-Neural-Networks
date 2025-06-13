import neat
import numpy as np
import time
import multiprocessing
import time
from stats import RLStatsCollector
from cartPole import *
from cart.customIzGenome import CustomIZGenome
from rate_iznn import RateIZNN


def encode_input(state, min_vals, max_vals, I_min=0, I_max=1):
    norm_state = (state - min_vals) / (max_vals - min_vals)
    I_values = I_min + norm_state * (I_max - I_min)
    return I_values

def decode_output(firing_rate, threshold=0.0):
    return 1 if firing_rate > threshold else 0

def simulate(genome, config):
    net = RateIZNN.create(genome, config)  
    state = np.array([0, 0, 0.05, 0])
    steps_balanced = 0

    while True:
        input_values = encode_input(state, min_vals, max_vals, 0, 1)
        net.set_inputs(input_values)

        output = net.advance(0.02)     
        #print(output)
        action = decode_output(output[0])
        state = simulate_cartpole(action, state)
        x, _, theta, _ = state
        
        if abs(x) > position_limit or abs(theta) > angle_limit:
            break
        
        steps_balanced += 1
        
        if steps_balanced >= 100000:
            break
    
        net.reset()
    
    return steps_balanced

def eval_single_genome(genome, config):
    genome.fitness = simulate(genome, config)
    return genome.fitness 

def gui(winner, config, generation_reached):
    state = np.array([0, 0, 0.05, 0])
    net = RateIZNN.create(winner, config)  
    running = True
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        input_values = encode_input(state, min_vals, max_vals, 0, 1)
        net.set_inputs(input_values)


        output = net.advance(0.02)     
        state = simulate_cartpole(output[0], state)
        x, _, theta, _ = state
        
        if abs(x) > position_limit or abs(theta) > angle_limit:
            net = RateIZNN.create(winner, config)  
            state = np.array([0, 0, 0.05, 0])
            time.sleep(1)

        net.reset() # Reset the network for the next iteration

        draw_cartpole(screen, state, generation_reached, 0, 0, "")

        clock.tick(50)
    pygame.quit()


def run(config_file, num_Generations=50):  
    start_time = time.time()
    config = neat.Config(CustomIZGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    def create_phenotype(genome):
        return RateIZNN.create(genome, config)

    neat.iznn.IZGenome.create_phenotype = create_phenotype
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

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_single_genome)
    winner = pop.run(pe.evaluate, num_Generations)

    elapsed_time = time.time() - start_time
    print(winner)
    print(f"STEPS : {winner.simulation_steps}")
    print(f"I MAX : {winner.input_scaling}")
    print(f"I MIN : {winner.input_min}")
    print(f"BACKGROUND : {winner.background}")
    print(f"Total time : {elapsed_time:.2f} sec")

    gui(winner, config , generation_reached)

def run_experiment(config_file, num_Generations=50):
    config = neat.Config(CustomIZGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    fitness_threshold = config.fitness_threshold if hasattr(config, "fitness_threshold") else 100000
    stats_collector = RLStatsCollector(fitness_threshold=fitness_threshold)
    stats_collector.start_experiment()

    def create_phenotype(genome):
        return RateIZNN.create(genome, config)
    neat.iznn.IZGenome.create_phenotype = create_phenotype

    pop = neat.population.Population(config)
    generation_reached = 0

    class GenerationReporter(neat.reporting.BaseReporter):
        def start_generation(self, generation):
            nonlocal generation_reached
            generation_reached = generation
            self.generation_start_time = time.time()

        def post_evaluate(self, config, population, species, best_genome):
            gen_time = time.time() - self.generation_start_time
            stats_collector.record_generation(best_genome.fitness, gen_time)

    pop.add_reporter(GenerationReporter())
    pop.add_reporter(neat.StdOutReporter(False))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_single_genome)
    winner = pop.run(pe.evaluate, num_Generations)
    stats_collector.end_experiment()

    # Evaluate winner average time to take an action
    state = np.array([0, 0, 0.05, 0])
    net = RateIZNN.create(winner, config)
    for _ in range(1000):
        input_values = encode_input(state, min_vals, max_vals, 0, 1)
        net.set_inputs(input_values)
        t0 = time.time()
        output = net.advance(0.02)
        action = decode_output(output[0])
        t1 = time.time()
        stats_collector.record_action_time(t1 - t0)
        state = simulate_cartpole(action, state)
        x, _, theta, _ = state
        if abs(x) > position_limit or abs(theta) > angle_limit:
            break

    return {
        "learning_time": stats_collector.learning_time(),
        "success": 1 if stats_collector.success_generation else 0,
        "success_generation": stats_collector.success_generation if stats_collector.success_generation else num_Generations,
        "mean_decision_time": stats_collector.mean_decision_time(),
        "best_fitness": max(stats_collector.fitness_history) if stats_collector.fitness_history else 0,
        "mean_fitness": np.mean(stats_collector.fitness_history) if stats_collector.fitness_history else 0,
        "hidden_nodes": len(winner.nodes) - len(config.genome_config.output_keys),
        "total_generations": len(stats_collector.fitness_history),
        "winner_steps": getattr(winner, "simulation_steps", None),
        "winner_input_scaling": getattr(winner, "input_scaling", None),
        "winner_input_min": getattr(winner, "input_min", None),
        "winner_background": getattr(winner, "background", None)
    }


if __name__ == "__main__":
    print("running Snn experiment")
    #run("cart/cartSnn_config.txt", 100)

    