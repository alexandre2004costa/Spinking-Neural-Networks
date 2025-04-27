from neat_iznn import *
from cartPole import *
from rate_iznn import RateIZNN
import gymnasium as gym
import numpy as np
import pygame
import time


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
