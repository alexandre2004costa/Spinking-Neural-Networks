import gymnasium as gym
import numpy as np
from neat_iznn import *
import time


def simulate(I_min, I_diff, I_background, genome, config, num_trials=15):
    trials_reward = []
    
    for _ in range(num_trials):
        net = RateIZNN.create(genome, config)  
        env = gym.make("LunarLander-v3", render_mode=None)
        state, _ = env.reset()
        
        total_reward = 0
        steps = 0
        done = False
          

        while not done:
            input_values = encode_input(state, env.observation_space.low, 
                                        env.observation_space.high, 0, 1)
            net.set_inputs(input_values)

            spike_counts = net.advance(0.02)
            # print(spike_counts)
            action = np.argmax(spike_counts)  

            state, reward, terminated, truncated, _ = env.step(action)
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
    env = gym.make("LunarLander-v3", render_mode="human")
    state, _ = env.reset()
    net = RateIZNN.create(winner, config)
    
    episode = 0
    steps = 0
    total_reward = 0
    
    while episode < 5:
        input_values = encode_input(state, env.observation_space.low, 
                                        env.observation_space.high, 0, 1)
        net.set_inputs(input_values)

        spike_counts = net.advance(0.2)
        action = np.argmax(spike_counts)  

        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated or steps >= 1000
        
        env.render()
        if hasattr(env, 'window') and hasattr(env.window, 'window_surface_v2'):
            text = f"Gen: {generation_reached}, Ep: {episode}, Steps: {steps}, Reward: {total_reward:.1f}"
        
        if terminated or truncated or steps >= 1000:
            episode += 1
            steps = 0
            total_reward = 0
            state, _ = env.reset()
            net = RateIZNN.create(winner, config)
            time.sleep(1)
    
    env.close()

config_values = {'I_min': -185.0, 
                'I_diff': 471, 
                'background': 10.0, 
                'weight_init_mean': 15.0,
                'weight_init_stdev': 2.7, 
                'weight_max_value': 48.0,
                'weight_min_value': -53.0, 
                'weight_mutate_power': 1.3, 
                'weight_mutate_rate': 0.76,
                'weight_replace_rate': 0.14}

run(config_values, simulate, "lunar/lunar_config_snn.txt", gui, 100)
