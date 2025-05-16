import neat
import numpy as np
import time
from rate_iznn import RateIZNN
import multiprocessing
from cartPole import *
from customIzGenome import CustomIZGenome
import time

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

def gui(winner, config, I_min, I_diff, I_background, generation_reached):
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

def run(config_values, config_file, num_Generations=50):  
    start_time = time.time()
    config = neat.Config(CustomIZGenome, neat.DefaultReproduction,
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

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_single_genome)
    winner = pop.run(pe.evaluate, num_Generations)

    elapsed_time = time.time() - start_time
    print(winner)
    print(f"STEPS : {winner.simulation_steps}")
    print(f"I MAX : {winner.input_scaling}")
    print(f"I MIN : {winner.input_min}")
    print(f"BACKGROUND : {winner.background}")
    print(f"Total time : {elapsed_time:.2f} sec")
    
    #gui(winner, config, config_values["I_min"], config_values["I_diff"], config_values["background"], generation_reached)
    


if __name__ == "__main__":
    run({'I_min': -185.20966099570762, 'I_diff': 471, 'background': 50.3531840776152606,'weight_init_mean': 2.0,
    'weight_init_stdev': 2.0,
    'weight_max_value': 30.0,
    'weight_min_value': -20.0,
    'weight_mutate_power': 2.0,
    'weight_mutate_rate': 0.76,
    'weight_replace_rate': 0.2}
        , "cart/cartSnn_config.txt", 50)
    