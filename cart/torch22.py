import multiprocessing
import os
import random
import neat
import visualize
import numpy as np
from cartPole import *
from neat.graphs import required_for_output   
import time
from multiprocessing import cpu_count
import snntorch as snn
import torch
import torch.nn as nn

class SpikingNetwork(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()
        
        beta = 0.85  
        threshold = 1.0  
        reset_mechanism = "subtract" 
        
        # Define as camadas
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism=reset_mechanism)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism=reset_mechanism)

    def forward(self, x, s1, s2):
        cur1 = self.fc1(x)
        spk1, s1 = self.lif1(cur1, s1)
        cur2 = self.fc2(spk1)
        spk2, s2 = self.lif2(cur2, s2)
        return spk2, s1, s2

def convert_genome_to_snn(genome, config):
    genome_config = config.genome_config
    required = required_for_output(genome_config.input_keys, genome_config.output_keys, genome.connections)
    
    num_inputs = len(genome_config.input_keys)
    num_outputs = len(genome_config.output_keys)
    num_hidden = len([n for n in required if n not in genome_config.input_keys and n not in genome_config.output_keys])

    
    snn_net = SpikingNetwork(num_inputs, num_hidden, num_outputs)
    
    with torch.no_grad():
        snn_net.fc1.weight.zero_()
        snn_net.fc2.weight.zero_()
        
        for cg in genome.connections.values():
            if not cg.enabled:
                continue

            i, o = cg.key
            if o not in required and i not in required:
                continue

        
            if i in genome_config.input_keys and o not in genome_config.output_keys:
                input_idx = genome_config.input_keys.index(i)
                hidden_idx = list(required).index(o) - num_inputs
                if 0 <= hidden_idx < num_hidden and 0 <= input_idx < num_inputs:
                    snn_net.fc1.weight[hidden_idx, input_idx] = torch.tensor(cg.weight)
            
            elif i not in genome_config.input_keys and o in genome_config.output_keys:
                hidden_idx = list(required).index(i) - num_inputs
                output_idx = genome_config.output_keys.index(o)
                if 0 <= hidden_idx < num_hidden and 0 <= output_idx < num_outputs:
                    snn_net.fc2.weight[output_idx, hidden_idx] = torch.tensor(cg.weight)
            
            elif i in genome_config.input_keys and o in genome_config.output_keys:
                input_idx = genome_config.input_keys.index(i)
                output_idx = genome_config.output_keys.index(o)
                if num_hidden > 0:  
                    snn_net.fc1.weight[0, input_idx] = torch.tensor(cg.weight)
                    snn_net.fc2.weight[output_idx, 0] = torch.tensor(1.0)
        
        for node_key in required:
            ng = genome.nodes[node_key]
            if node_key not in genome_config.input_keys and node_key not in genome_config.output_keys:
                idx = list(required).index(node_key) - num_inputs
                if 0 <= idx < num_hidden:
                    snn_net.lif1.beta = torch.tensor(ng.a)
            elif node_key in genome_config.output_keys:
                idx = genome_config.output_keys.index(node_key)
                if 0 <= idx < num_outputs:
                    snn_net.lif2.beta = torch.tensor(ng.a)
    
    return snn_net

def encode_input(state, min_vals, max_vals, I_min=70.0, I_max=100.0):
    norm_state = (state - min_vals) / (max_vals - min_vals)
    I_values = I_min + norm_state * (I_max - I_min)
    return I_values

def simulate(I_min, I_diff, genome, config):
    net = convert_genome_to_snn(genome, config)
    state = np.array([0, 0, 0.05, 0])
    total_reward = 0
    steps_balanced = 0
    

    
    num_hidden = net.fc1.out_features
    num_outputs = net.fc2.out_features
    s1 = torch.zeros(1, num_hidden)
    s2 = torch.zeros(1, num_outputs)
    
    while True:
        input_values = torch.tensor(encode_input(state, min_vals, max_vals, I_min, I_min + I_diff), 
                                  dtype=torch.float32).unsqueeze(0)
        
        cur1 = net.fc1(input_values)
        spk1, s1 = net.lif1(cur1, s1)
        cur2 = net.fc2(spk1)
        spk2, s2 = net.lif2(cur2, s2)
        
        if spk2.sum().item() == 0 :
            action = 0
        else:
            action = 1
        
        state = simulate_cartpole(action, state)
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


param_ranges = {
    "pop_size": (50, 200),
    "bias_mutate_rate": (0.5, 1.0),
    "weight_mutate_rate": (0.5, 1.5),
    "conn_add_prob": (0.1, 0.5),
    "conn_delete_prob": (0.1, 0.5),
    "node_add_prob": (0.05, 0.3),
    "node_delete_prob": (0.05, 0.3),
    "compatibility_disjoint_coefficient": (0.5, 2.0),
    "compatibility_weight_coefficient": (0.2, 1.0),
    "max_stagnation": (20, 100),
    "I_min": (-200.0, -150.0),
    "I_diff": (400, 500),
}

def random_config():
    config_values = {}
    for param, (min_val, max_val) in param_ranges.items():
        if isinstance(min_val, float):
            config_values[param] = random.uniform(min_val, max_val)
        else:
            config_values[param] = random.randint(min_val, max_val)
    return config_values

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
    #config.genome_config.num_hidden = int(config_values["num_hidden"])

    pop = neat.population.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)


    def eval_genome_wrapper(genomes, config):
        eval_genome(config_values["I_min"], config_values["I_diff"], genomes, config)
    
    winner = pop.run(eval_genome_wrapper, 100)
    
    print('\nBest genome:\n', winner)
    print('I-min:', config_values["I_min"])
    print('I-diff:', config_values["I_diff"])

    if winner.fitness >= 100000:

        # Display the best genome
        state = np.array([0, 0, 0.05, 0])
        net = convert_genome_to_snn(winner, config)
        running = True
        pygame.init()
        screen = pygame.display.set_mode((screen_width, screen_height))
        clock = pygame.time.Clock()


        num_hidden = net.fc1.out_features
        num_outputs = net.fc2.out_features
        s1 = torch.zeros(1, num_hidden)
        s2 = torch.zeros(1, num_outputs)
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            input_values = torch.tensor(encode_input(state, min_vals, max_vals, config_values["I_min"], config_values["I_min"] + config_values["I_diff"]), 
                                    dtype=torch.float32).unsqueeze(0)
            
            cur1 = net.fc1(input_values)
            spk1, s1 = net.lif1(cur1, s1)
            cur2 = net.fc2(spk1)
            spk2, s2 = net.lif2(cur2, s2)
            
            if spk2.sum().item() == 0 :
                action = 0
            else:
                action = 1
            
            state = simulate_cartpole(action, state)
            x, _, theta, _ = state
            message = ""
            if abs(x) > position_limit or abs(theta) > angle_limit:
                message = "Failed! Restarting..."
                state = np.array([0, 0, 0.05, 0])
                net = convert_genome_to_snn(winner, config)
                num_hidden = net.fc1.out_features
                num_outputs = net.fc2.out_features
                s1 = torch.zeros(1, num_hidden)
                s2 = torch.zeros(1, num_outputs)
                time.sleep(1)

            draw_cartpole(screen, state, 0, 0, 0, message)

            clock.tick(50)
        pygame.quit()
    return winner.fitness if winner else 0


def random_search(n_trials=10):
    best_config = None
    best_fitness = float('-inf')

    for _ in range(n_trials):
        config_values = {'pop_size': 171, 'bias_mutate_rate': 0.5230790860447896, 'weight_mutate_rate': 1.2258341513572386, 'conn_add_prob': 0.3476145006301879, 'conn_delete_prob': 0.18904010926789036, 'node_add_prob': 0.2374980918364757, 'node_delete_prob': 0.15772024546933505, 'compatibility_disjoint_coefficient': 1.943134293658721, 'compatibility_weight_coefficient': 0.9148384084033248, 'max_stagnation': 27, 'I_min': -184.75081947146862, 'I_diff': 469}
        fitness = run(config_values)
        if fitness > best_fitness:
            best_fitness = fitness
            best_config = config_values
            
        print(f"Test with config {config_values}, Fitness: {fitness}")
    
    print(f"Best configuration: {best_config}, Fitness: {best_fitness}")
    return best_config

if __name__ == '__main__':
    best_config = random_search(1)
