from cart.cartSnn import *
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
    env = gym.make("CartPole-v1")
    state, _ = env.reset()
    steps_balanced = 0

    while True:
        input_values = encode_input(state,env.observation_space.low, 
                                        env.observation_space.high, 0, 1)
        net.set_inputs(input_values)
        output = net.advance(0.02)     
        #print(output)
        action = decode_output(output[0])
        #print(action)
        state, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
        
        steps_balanced += 1
        
        if steps_balanced >= 100000:
            break

    env.close()
    return steps_balanced

def gui(winner, config, I_min, I_diff, I_background, generation_reached):
    env = gym.make("CartPole-v1", render_mode="human")
    net = RateIZNN.create(winner, config)  
    state, _ = env.reset()

    done = False
    while not done:
        # Processa o estado
        input_values = encode_input(state,env.observation_space.low, 
                                        env.observation_space.high, 0, 1)
        net.set_inputs(input_values)
        output = net.advance(0.02)
        action = decode_output(output[0])
        state, reward, terminated, truncated, _ = env.step(action)
        
        if terminated or truncated:
            time.sleep(1)
            state, _ = env.reset()
            net = RateIZNN.create(winner, config)
        
        time.sleep(0.02)
    env.close()

run({'I_min': -185.20966099570762, 'I_diff': 471, 'background': 50.3531840776152606,'weight_init_mean': 18.0,
'weight_init_stdev': 4.0,
'weight_max_value': 60.0,
'weight_min_value': -40.0,
'weight_mutate_power': 2.0,
'weight_mutate_rate': 0.76,
'weight_replace_rate': 0.2}
    , simulate, "cart/cartSnn_config.txt", gui, 50)