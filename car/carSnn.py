import gymnasium as gym
import numpy as np
from neat_iznn import *
import time

def simulate(I_min, I_diff, I_background, genome, config, num_trials=5):
    trials_reward = []
    
    for _ in range(num_trials):
        net = neat.iznn.IZNN.create(genome, config)  
        env = gym.make("MountainCarContinuous-v0", render_mode=None)
        state, _ = env.reset()
        
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            input_values = encode_input(state, env.observation_space.low, 
                                    env.observation_space.high, I_min, I_min + I_diff)
            net.set_inputs(input_values)

            for neuron in net.neurons.values():
                neuron.current += I_background
            for value in net.input_values:
                value += I_background

            output = net.advance(0.02)
            action = np.clip(output[0], -1.0, 1.0)  
            
            state, reward, terminated, truncated, _ = env.step([action])  
            
            total_reward += reward
            steps += 1
            done = terminated or truncated or steps >= 1000
        
        env.close()
        trials_reward.append(float(total_reward))
    
    avg_reward = sum(trials_reward) / num_trials
    if np.isnan(avg_reward) or np.isinf(avg_reward):
        return 0.0
    
    return avg_reward

def gui(winner, config, I_min, I_diff, I_background, generation_reached):
    env = gym.make("MountainCarContinuous-v0", render_mode="human")
    state, _ = env.reset()
    net = neat.iznn.IZNN.create(winner, config)
    
    episode = 0
    steps = 0
    
    while episode < 5:
        input_values = encode_input(state, env.observation_space.low, 
                                env.observation_space.high, I_min, I_min + I_diff)
        net.set_inputs(input_values)

        for neuron in net.neurons.values():
            neuron.v += I_background

        output = net.advance(0.02)
        action = np.clip(output[0], -1.0, 1.0)
        
        state, _, terminated, truncated, _ = env.step([action])
        steps += 1
        
        env.render()
        if hasattr(env, 'window') and hasattr(env.window, 'window_surface_v2'):
            text = f"Generation: {generation_reached}, Episode: {episode}, Steps: {steps}"
            position, velocity = state
            text += f"\nPos: {position:.2f}, Vel: {velocity:.2f}, Action: {action:.2f}"
        
        if terminated or truncated or steps >= 1000:
            episode += 1
            steps = 0
            state, _ = env.reset()
            net = neat.iznn.IZNN.create(winner, config)
            time.sleep(1)
    
    env.close()

config_values = {'I_min': -185.20966099570762, 
                'I_diff': 471, 
                'background': 50.3531840776152606, 
                'weight_init_mean': 15.312074270957652,
                'weight_init_stdev': 2.721486079549555, 
                'weight_max_value': 48.37166968093461,
                'weight_min_value': -53.90179586039318, 
                'weight_mutate_power': 1.348883789152442, 
                'weight_mutate_rate': 0.7612022202685518, 
                'weight_replace_rate': 0.1415782630854031}

run(config_values, simulate, "car/mountain_config_snn.txt", gui, 100)
