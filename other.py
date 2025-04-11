import numpy as np
import neat
import math
import pygame
from cartPole import *
import snntorch as snn
from neat.graphs import required_for_output  
import torch
import torch.nn as nn
import numpy as np
import neat
import math
import pygame
from cartPole import *


generation_number = 0

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
    
def convert_default_genome_to_snn(genome, config):
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

            # input -> hidden connection
            if i in genome_config.input_keys and o not in genome_config.output_keys:
                input_idx = genome_config.input_keys.index(i)
                hidden_idx = list(required).index(o) - num_inputs
                if 0 <= hidden_idx < num_hidden and 0 <= input_idx < num_inputs:
                    snn_net.fc1.weight[hidden_idx, input_idx] = torch.tensor(cg.weight)
            
            # hidden -> output connection
            elif i not in genome_config.input_keys and o in genome_config.output_keys:
                hidden_idx = list(required).index(i) - num_inputs
                output_idx = genome_config.output_keys.index(o)
                if 0 <= hidden_idx < num_hidden and 0 <= output_idx < num_outputs:
                    snn_net.fc2.weight[output_idx, hidden_idx] = torch.tensor(cg.weight)
            
            # output -> hidden connection (not typical in SNNs, but included for completeness)
            elif i in genome_config.input_keys and o in genome_config.output_keys:
                input_idx = genome_config.input_keys.index(i)
                output_idx = genome_config.output_keys.index(o)
                if num_hidden > 0:
                    snn_net.fc1.weight[0, input_idx] = torch.tensor(cg.weight)
                    snn_net.fc2.weight[output_idx, 0] = torch.tensor(1.0)
        
        # bias
        for node_key in required:
            if node_key not in genome_config.input_keys:  # Skip input nodes
                node = genome.nodes[node_key]
                if node_key not in genome_config.output_keys:
                    # Hidden neuron
                    idx = list(required).index(node_key) - num_inputs
                    if 0 <= idx < num_hidden:
                        snn_net.lif1.threshold = torch.tensor(node.bias)  
                else:
                    # Output neuron
                    idx = genome_config.output_keys.index(node_key)
                    if 0 <= idx < num_outputs:
                        snn_net.lif2.threshold = torch.tensor(node.bias)  
    
    return snn_net

def encode_input(state, min_vals, max_vals, I_min=70.0, I_max=100.0):
    norm_state = (state - min_vals) / (max_vals - min_vals)
    I_values = I_min + norm_state * (I_max - I_min)
    return I_values

def eval_genomes(genomes, config):
    global generation_number
    generation_number += 1
    max_fitness = 0
    for _, genome in genomes:
        net = convert_default_genome_to_snn(genome, config)
        num_hidden = net.fc1.out_features
        num_outputs = net.fc2.out_features
        s1 = torch.zeros(1, num_hidden)
        s2 = torch.zeros(1, num_outputs)
        fitness = 0
        state = np.array([0, 0, 0.05, 0])
        while True:
            input_values = torch.tensor(encode_input(state, min_vals, max_vals, -158, 480), 
                                  dtype=torch.float32).unsqueeze(0)
            #print(input_values)

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
            #fitness += (math.cos(theta) + 1)
            fitness += 1
            if fitness >= 100000:
                break
        genome.fitness = fitness
        max_fitness = max(max_fitness, fitness)

def run_neat(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(eval_genomes, 100)
    print('\nBest genome:\n', winner)

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
        outputs = net.activate(state) 
        action = np.argmax(outputs)
        state = simulate_cartpole(action, state)
        x, _, theta, _ = state

        message = ""
        if abs(x) > position_limit or abs(theta) > angle_limit:
            message = "Failed! Restarting..."
            state = np.array([0, 0, 0.05, 0])

        draw_cartpole(screen, state, generation_number, 0, 0, message)

        clock.tick(50)
    pygame.quit()

if __name__ == '__main__':
    config_path = 'config-feedforward2.txt'
    run_neat(config_path)