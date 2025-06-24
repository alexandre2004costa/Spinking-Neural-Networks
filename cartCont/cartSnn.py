import neat
import numpy as np
import time
import multiprocessing
import time
from stats import RLStatsCollector
from cartPole import *
from cartCont.customIzGenome import CustomIZGenome
from rate_iznn_cont import RateIZNN

min_vals = np.array([-2.4, -0.21])
max_vals = np.array([2.4, 0.21])

def encode_input(state, min_vals, max_vals, I_min=0, I_max=1):
    norm_state = (state - min_vals) / (max_vals - min_vals)
    I_values = I_min + norm_state * (I_max - I_min)
    return I_values

def compute_force(weighted_sum, sigma=1.0):
    Fn = 1.0 / (1.0 + np.exp(-sigma * weighted_sum))
    Ft = 10 * (2 * Fn - 1)
    #print(f"WS : {weighted_sum} Fn: {Fn}, Ft: {Ft}")
    return Ft

def simulate(genome, config):
    net = RateIZNN.create(genome, config)  
    state = np.array([0, 0, 0.05, 0])
    steps_balanced = 0

    while True:
        reduced_state = np.array([state[0], state[2]])
        input_values = encode_input(reduced_state, min_vals, max_vals, 0, 1)
        net.set_inputs(input_values)

        output = net.advance(0.01)     
        #print(output)

        action = compute_force(output, genome.sigma)
        #print(action)
        state = simulate_cartpole_cont(action, state)
        x, _, theta, _ = state
        
        if abs(x) > position_limit or abs(theta) > angle_limit:
            break
        
        steps_balanced += 1
        
        if steps_balanced >= 5000:
            break
    
        #net.reset()
    
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


        output = net.advance(0.01)     
        state = simulate_cartpole_cont(output[0], state)
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
    print(f"Sigma : {winner.sigma}")
    print(f"Total time : {elapsed_time:.2f} sec")
    
    #gui(winner, config, config_values["I_min"], config_values["I_diff"], config_values["background"], generation_reached)
    


if __name__ == "__main__":
    run("cartCont/cartSnn_config.txt", 350)
    