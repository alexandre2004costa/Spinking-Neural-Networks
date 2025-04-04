import multiprocessing
import os
import random
import neat
import numpy as np
from cartPole import *
from iznn import IZGenome, IZNN

generation_number = 0
dt = 0.02
maxFitness = 0

def encode_input(state, min_vals, max_vals, I_min=70.0, I_max=100.0):
    norm_state = (state - min_vals) / (max_vals - min_vals)
    I_values = I_min + norm_state * (I_max - I_min)
    return I_values

def simulate(genome, config):
    if isinstance(genome, (list, tuple)):
        genome = genome[1]
    
    try:
        net = IZNN.create(genome, config)
        state = np.array([0, 0, 0.05, 0])
        total_reward = 0
        steps_balanced = 0
        max_steps = 100000

        # Debug: Print network structure and initial state
        print("\nNetwork structure:")
        output_neuron = net.neurons[0]
        hidden_inputs = [i for i, _ in output_neuron.inputs if i > 0]
        if hidden_inputs:
            print(f"Output receives from hidden nodes: {hidden_inputs}")
            print(f"Output neuron initial state - v: {output_neuron.v:.2f}, u: {output_neuron.u:.2f}")
        
        while steps_balanced < max_steps:
            try:
                input_values = encode_input(state, min_vals, max_vals)
                output = net.activate(input_values)
                action = 1 if output[0] > 0 else 0
                
                # Debug every 1000 steps
                if steps_balanced % 1000 == 0:
                    print(f"Step {steps_balanced}: v={output_neuron.v:.2f}, u={output_neuron.u:.2f}, current={output_neuron.current:.2f}")
                
                state = simulate_cartpole(action, state)
                x, _, theta, _ = state
                
                if abs(x) > position_limit or abs(theta) > angle_limit:
                    print(f"Failed at step {steps_balanced} - x: {x:.2f}, theta: {theta:.2f}")
                    break
                
                total_reward += (math.cos(theta) + 1)
                steps_balanced += 1
                
            except Exception as e:
                print(f"Error during simulation step {steps_balanced}: {str(e)}")
                raise
        
        return total_reward
        
    except Exception as e:
        print(f"Critical error in simulate: {str(e)}")
        print(f"Genome: {genome}")
        return 0.0

def eval_genomes(genomes, config):
    global generation_number
    generation_number += 1
    print(f"\nGeneration {generation_number}")
    
    for genome_id, genome in genomes:
        try:
            fitness = simulate(genome, config)
            genome.fitness = fitness
            print(f"Genome {genome_id} fitness: {fitness}")
        except Exception as e:
            print(f"Error evaluating genome {genome_id}: {str(e)}")
            genome.fitness = 0.0

param_ranges = {
    "pop_size": (100, 200),
    "bias_mutate_rate": (0.5, 1.0),
    "weight_mutate_rate": (0.5, 1.0),
    "conn_add_prob": (0.1, 0.5),
    "conn_delete_prob": (0.1, 0.5),
    "node_add_prob": (0.05, 0.3),
    "node_delete_prob": (0.05, 0.3),
    "compatibility_disjoint_coefficient": (0.5, 2.0),
    "compatibility_weight_coefficient": (0.2, 1.0),
    "max_stagnation": (20, 100),
}

def random_config():
    return {param: random.uniform(*param_ranges[param]) if isinstance(param_ranges[param][0], float) 
            else random.randint(*param_ranges[param]) for param in param_ranges}

def run_neat_with_config(config_values):
    config = neat.Config(IZGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-spiking.txt')
    
    #config.pop_size = int(config_values["pop_size"])
    #config.genome_config.bias_mutate_rate = config_values["bias_mutate_rate"]
    #config.genome_config.weight_mutate_rate = config_values["weight_mutate_rate"]
    #config.genome_config.conn_add_prob = config_values["conn_add_prob"]
    #config.genome_config.conn_delete_prob = config_values["conn_delete_prob"]
    #config.genome_config.node_add_prob = config_values["node_add_prob"]
    #config.genome_config.node_delete_prob = config_values["node_delete_prob"]
    #config.species_set_config.compatibility_disjoint_coefficient = config_values["compatibility_disjoint_coefficient"]
    #config.species_set_config.compatibility_weight_coefficient = config_values["compatibility_weight_coefficient"]
    #config.stagnation_config.max_stagnation = int(config_values["max_stagnation"])

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    #try:
    winner = pop.run(eval_genomes, 100)
    return winner.fitness if winner else 0.0
    #except Exception as e:
        #print(f"Error in evolution: {e}")
        #return 0.0

def random_search(n_trials=10):
    best_config = None
    best_fitness = float('-inf')

    for _ in range(n_trials):
        config_values = random_config()
        fitness = run_neat_with_config(config_values)
        if fitness > best_fitness:
            best_fitness = fitness
            best_config = config_values
            
        print(f"Test with config {config_values}, Fitness: {fitness}, dt {dt}")
    
    print(f"Best configuration: {best_config}, Fitness: {best_fitness}")
    return best_config

if __name__ == '__main__':
    best_config = random_search(1)