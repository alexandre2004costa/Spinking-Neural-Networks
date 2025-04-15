import multiprocessing
import os
import random
import neat
import visualize
import numpy as np
from cartPole import *
import time
from multiprocessing import cpu_count
import optuna

def encode_input(state, min_vals, max_vals, I_min=20.0, I_max=100.0):
    norm_state = (state - min_vals) / (max_vals - min_vals)
    I_values = I_min + norm_state * (I_max - I_min)
    return I_values

def simulate(I_min, I_diff, I_background, genome, config):
    net = neat.iznn.IZNN.create(genome, config)  
    state = np.array([0, 0, 0.05, 0])
    steps_balanced = 0

    
    while True:
        input_values = encode_input(state, min_vals, max_vals, I_min, I_min + I_diff)
        net.set_inputs(input_values)

        for neuron in net.neurons.values():
            neuron.current += I_background
        for value in net.input_values:
            value += I_background

        output = net.advance(0.02)
        #print(output[0])        
        state = simulate_cartpole(output[0], state)
        x, _, theta, _ = state
        
        if abs(x) > position_limit or abs(theta) > angle_limit:
            break
        
        steps_balanced += 1
        
        if steps_balanced >= 100000:
            break
    
    return steps_balanced


def eval_genome(I_min, I_diff, background, genomes, config):
    for _, genome in genomes:
        genome.fitness = simulate(I_min, I_diff, background, genome, config)

def run(config_values):

    config = neat.Config(neat.iznn.IZGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "oi.txt")
    
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

    config.genome_config.weight_init_mean = config_values["weight_init_mean"]
    config.genome_config.weight_init_stdev = config_values["weight_init_stdev"]
    config.genome_config.weight_max_value = config_values["weight_max_value"]
    config.genome_config.weight_min_value = config_values["weight_min_value"]
    config.genome_config.weight_mutate_power = config_values["weight_mutate_power"]
    config.genome_config.weight_mutate_rate = config_values["weight_mutate_rate"]
    config.genome_config.weight_replace_rate = config_values["weight_replace_rate"]

    pop = neat.population.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)


    def eval_genome_wrapper(genomes, config):
        eval_genome(config_values["I_min"], config_values["I_diff"], config_values["background"],  genomes, config)
    
    winner = pop.run(eval_genome_wrapper, 50)
    print(winner)
    return winner.fitness if winner else 0


def create_study():
    def objective(trial):
        config_values = {
            #"pop_size": trial.suggest_int("pop_size", 150, 250),
            #"bias_mutate_rate": trial.suggest_float("bias_mutate_rate", 0.4, 0.7),
            #"weight_mutate_rate": trial.suggest_float("weight_mutate_rate", 0.8, 1.1),
            #"conn_add_prob": trial.suggest_float("conn_add_prob", 0.1, 0.2),
            #"conn_delete_prob": trial.suggest_float("conn_delete_prob", 0.3, 0.4),
            #"node_add_prob": trial.suggest_float("node_add_prob", 0.1, 0.2),
            #"node_delete_prob": trial.suggest_float("node_delete_prob", 0.2, 0.3),
            #"compatibility_disjoint_coefficient": trial.suggest_float("compatibility_disjoint_coefficient", 0.8, 1.0),
            #"compatibility_weight_coefficient": trial.suggest_float("compatibility_weight_coefficient", 0.6, 0.8),
            #"max_stagnation": trial.suggest_int("max_stagnation", 30, 40),
            "I_min": trial.suggest_float("I_min", -200, -150),
            "I_diff": trial.suggest_int("I_diff", 400, 500),
            "background": trial.suggest_float("background", 0.0, 20.0),
            "weight_init_mean": trial.suggest_float("weight_init_mean", 15.0, 25.0),      # Center around 20
            "weight_init_stdev": trial.suggest_float("weight_init_stdev", 1.0, 3.0),      # Vary around 2
            "weight_max_value": trial.suggest_float("weight_max_value", 40.0, 60.0),      # Vary around 50
            "weight_min_value": trial.suggest_float("weight_min_value", -60.0, -40.0),    # Symmetric with max
            "weight_mutate_power": trial.suggest_float("weight_mutate_power", 0.5, 1.5),  # Around 1.0
            "weight_mutate_rate": trial.suggest_float("weight_mutate_rate", 0.6, 1.0),    # High mutation rate
            "weight_replace_rate": trial.suggest_float("weight_replace_rate", 0.05, 0.15)  # Around 0.1
        }
        
        n_runs = 1
        total_fitness = 0
        for _ in range(n_runs):
            fitness = run(config_values)
            total_fitness += fitness
        
        return total_fitness / n_runs

    study = optuna.create_study(direction="maximize")
    
    def print_callback(study, trial):
        print(f"\nTrial {trial.number}:")
        print(f"Current value: {trial.value}")
        print("Best parameters:", study.best_params)
        print("Best value:", study.best_value)
    
    return study, objective, print_callback

def optimize_parameters(n_trials=50):
    """Run the optimization process"""
    study, objective, callback = create_study()
    
    study.optimize(objective, n_trials=n_trials, callbacks=[callback])
    
    print("\n=== Optimization Complete ===")
    print("Best parameters:", study.best_params)
    print("Best fitness:", study.best_value)
    
    return study.best_params

if __name__ == '__main__':

    #best_params = optimize_parameters(10)
    

    #print("\nTesting best parameters...")
    #final_fitness = run(best_params)
    #print(f"Final test fitness: {final_fitness}")
    #print(best_params)
    run({'I_min': -185.20966099570762, 'I_diff': 471, 'background': 50.3531840776152606, 'weight_init_mean': 15.312074270957652, 'weight_init_stdev': 2.721486079549555, 'weight_max_value': 48.37166968093461, 'weight_min_value': -53.90179586039318, 'weight_mutate_power': 1.348883789152442, 'weight_mutate_rate': 0.7612022202685518, 'weight_replace_rate': 0.1415782630854031})