import gymnasium as gym
import numpy as np
import neat
import time
import multiprocessing

def simulate(genome, config, num_trials=5):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    trials_reward = []
    
    for trial in range(num_trials):
        env = gym.make("LunarLander-v3", render_mode=None)
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            output = net.activate(state)  
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
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    episode = 0
    steps = 0
    total_reward = 0
    
    while episode < 5:
        output = net.activate(state)
        action = np.argmax(output)
        
        state, reward, terminated, truncated, _ = env.step(action)
        steps += 1
        total_reward += reward
        
        env.render()
        if hasattr(env, 'window') and hasattr(env.window, 'window_surface_v2'):
            text = f"Generation: {generation_reached}, Episode: {episode}"
            text += f"\nSteps: {steps}, Reward: {total_reward:.2f}"
        
        if terminated or truncated or steps >= 1000:
            episode += 1
            steps = 0
            total_reward = 0
            state, _ = env.reset()
            time.sleep(1)
    
    env.close()

def eval_genome_parallel(genome, config):
    return simulate(genome, config)

def eval_genomes(genomes, config):
    for _, genome in genomes:
        genome.fitness = simulate(genome, config)

def run_neat(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_file)
    
    pop = neat.Population(config)
    
    generation_reached = 0
    class GenerationReporter(neat.reporting.BaseReporter):
        def start_generation(self, generation):
            nonlocal generation_reached
            generation_reached = generation
    
    pop.add_reporter(GenerationReporter())
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome_parallel)
    
    winner = pop.run(pe.evaluate, 1000)
    
    #print('\nBest genome:\n{!s}'.format(winner))
    
    gui(winner, config, generation_reached)
    
if __name__ == '__main__':
    run_neat("lunar/lunar_config_ann.txt")