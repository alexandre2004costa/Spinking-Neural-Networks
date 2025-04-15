import gymnasium as gym
import numpy as np
import pygame
from cart.neat_iznn import *
import time

def simulate(I_min, I_diff, I_background, genome, config):
    net = neat.iznn.IZNN.create(genome, config)  
    env = gym.make("MountainCar-v0", render_mode=None)
    state, _ = env.reset()
    
    total_reward = 0
    steps = 0
    done = False
    
    while not done:
        # Normalize state and encode for SNN
        input_values = encode_input(state, env.observation_space.low, 
                                  env.observation_space.high, I_min, I_min + I_diff)
        net.set_inputs(input_values)

        # Add background current
        for neuron in net.neurons.values():
            neuron.current += I_background
        for value in net.input_values:
            value += I_background

        # Get network output and convert to action
        output = net.advance(0.02)
        action = np.argmax(output)  # Convert to discrete action (0, 1, 2)
        
        # Execute action
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        
        # Check if episode is done
        done = terminated or truncated or steps >= 1000
    
    env.close()
    return total_reward

def gui(winner, config, I_min, I_diff, I_background, generation_reached):
    env = gym.make("MountainCar-v0", render_mode="human")
    state, _ = env.reset()
    net = neat.iznn.IZNN.create(winner, config)
    
    done = False
    episode = 0
    steps = 0
    
    while episode < 5:  # Show 5 episodes
        input_values = encode_input(state, env.observation_space.low, 
                                  env.observation_space.high, I_min, I_min + I_diff)
        net.set_inputs(input_values)

        # Add background current
        for neuron in net.neurons.values():
            neuron.current += I_background
        for value in net.input_values:
            value += I_background

        output = net.advance(0.02)
        action = np.argmax(output)
        
        state, reward, terminated, truncated, _ = env.step(action)
        steps += 1
        
        # Show generation info
        env.render()
        if hasattr(env, 'window') and hasattr(env.window, 'window_surface_v2'):
            text = f"Generation: {generation_reached}, Episode: {episode}, Steps: {steps}"
            # Add text to render window (implementation depends on gym version)
        
        if terminated or truncated or steps >= 1000:
            episode += 1
            steps = 0
            state, _ = env.reset()
            net = neat.iznn.IZNN.create(winner, config)
            time.sleep(1)
    
    env.close()

# Configure NEAT parameters for MountainCar
config_values = {
    'I_min': -100.0,           # Adjusted for MountainCar
    'I_diff': 200.0,           # Adjusted for MountainCar
    'background': 20.0,        # Background current
    'weight_init_mean': 15.0,
    'weight_init_stdev': 2.0,
    'weight_max_value': 50.0,
    'weight_min_value': -50.0,
    'weight_mutate_power': 1.0,
    'weight_mutate_rate': 0.8,
    'weight_replace_rate': 0.1
}

# Run the evolution
run(config_values, simulate, "mountain_config.txt", gui, 100)
