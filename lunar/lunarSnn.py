import gymnasium as gym
import numpy as np
from neat_iznn import *
import time
import multiprocessing
import neat


def simulate(genome, config, num_trials=6):
    trials_reward = []
    
    for _ in range(num_trials):
        net = RateIZNN.create(genome, config)  
        env = gym.make("LunarLander-v3", render_mode=None)
        state, _ = env.reset()
        
        total_reward = 0
        steps = 0
        done = False
          

        while not done:
            input_values = encode_input(state, env.observation_space.low, env.observation_space.high)
            net.set_inputs(input_values)

            output = net.advance(0.02)
            #print(output)
            action = np.argmax(output)  

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

def gui(winner, config, generation_reached):
    env = gym.make("LunarLander-v3", render_mode="human")
    state, _ = env.reset()
    net = RateIZNN.create(winner, config)
    
    episode = 0
    steps = 0
    total_reward = 0
    
    while episode < 10:
        input_values = encode_input(state, env.observation_space.low, env.observation_space.high)
        net.set_inputs(input_values)

        spike_counts = net.advance(0.2)
        action = np.argmax(spike_counts)  

        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        
        env.render()
        if hasattr(env, 'window') and hasattr(env.window, 'window_surface_v2'):
            text = f"Generation: {generation_reached}, Episode: {episode}"
            text += f"\nSteps: {steps}, Reward: {total_reward:.2f}"
        
        if terminated or truncated or steps >= 1000:
            print(f"Episódio {episode+1} terminado com recompensa total: {total_reward:.2f}")
            episode += 1
            steps = 0
            total_reward = 0
            state, _ = env.reset()
            net = RateIZNN.create(winner, config)
            time.sleep(1)
    
    env.close()

def eval_single_genome(genome, config):
    genome.fitness = simulate(genome, config)
    return genome.fitness 

def run(config_file, num_Generations=50):  

    config = neat.Config(neat.iznn.IZGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    def create_phenotype(genome):
        return RateIZNN.create(genome, config)

    neat.iznn.IZGenome.create_phenotype = create_phenotype

    pop = neat.population.Population(config)
    generation_reached = 0

    class GenerationReporter(neat.reporting.BaseReporter):
        def start_generation(self, generation):
            nonlocal generation_reached
            generation_reached = generation

    pop.add_reporter(GenerationReporter())
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_single_genome)
    winner = pop.run(pe.evaluate, num_Generations)

    print(winner)
    gui(winner, config, generation_reached)

if __name__ == "__main__":
    run("lunar/lunar_config_snn.txt", 100)
