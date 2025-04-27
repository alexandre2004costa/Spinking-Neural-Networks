from neat_iznn import *
from cartPole import *
from rate_iznn import RateIZNN
import gymnasium as gym
import numpy as np
import pygame
import time


def decode_output(firing_rate, threshold=0.0):
    return 1 if firing_rate > threshold else 0

def simulate(I_min, I_diff, I_background, genome, config):
    net = RateIZNN.create(genome, config)  
    state = np.array([0, 0, 0.05, 0])
    steps_balanced = 0

    while True:
        input_values = encode_input(state, min_vals, max_vals, 0, 1)
        net.set_inputs(input_values)

        output = net.advance(0.02)     
        #print(output)
        action = decode_output(output[0])
        #print(action)
        state = simulate_cartpole(action, state)
        x, _, theta, _ = state
        
        if abs(x) > position_limit or abs(theta) > angle_limit:
            break
        
        steps_balanced += 1
        
        if steps_balanced >= 100000:
            break
    
    return steps_balanced

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

        for neuron in net.neurons.values():
            neuron.current += I_background
        for value in net.input_values:
            value += I_background

        output = net.advance(0.02)     
        state = simulate_cartpole(output[0], state)
        x, _, theta, _ = state
        
        if abs(x) > position_limit or abs(theta) > angle_limit:
            net = RateIZNN.create(winner, config)  
            state = np.array([0, 0, 0.05, 0])
            time.sleep(1)

        draw_cartpole(screen, state, generation_reached, 0, 0, "")

        clock.tick(50)
    pygame.quit()

run({'I_min': -185.20966099570762, 'I_diff': 471, 'background': 50.3531840776152606,'weight_init_mean': 18.0,
'weight_init_stdev': 4.0,
'weight_max_value': 60.0,
'weight_min_value': -40.0,
'weight_mutate_power': 2.0,
'weight_mutate_rate': 0.76,
'weight_replace_rate': 0.2}
    , simulate, "cart/cartSnn_config.txt", gui, 50)