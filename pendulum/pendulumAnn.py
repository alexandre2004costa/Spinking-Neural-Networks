import gymnasium as gym
import numpy as np
import neat
import time
import multiprocessing


    
def simulate(genome, config, num_trials=5):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    trials_reward = []

    for _ in range(num_trials):
        env = gym.make("Pendulum-v1", render_mode=None)
        state, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done:

            output = net.activate(state)
            # Output ∈ [-1, 1] -> [-2, 2]
            
            action = np.clip(output[0], -1.0, 1.0) * 2.0  
            state, reward, terminated, truncated, _ = env.step([action])
            total_reward += reward
            done = terminated or truncated or steps >= 200
            steps += 1

        env.close()
        trials_reward.append(total_reward)

    avg_reward = sum(trials_reward) / len(trials_reward)
    if np.isnan(avg_reward) or np.isinf(avg_reward):
        print("Genome produced NaN or Inf reward, returning -1000.0")
        return -1000.0
    return avg_reward

def gui(winner, config, generation_reached):
    env = gym.make("Pendulum-v1", render_mode="human")
    state, _ = env.reset()
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    episode = 0
    steps = 0
    max_episodes = 10
    total_reward = 0

    while episode < max_episodes:
        output = net.activate(state)
        # Output ∈ [-1, 1] -> [-2, 2]
        action = np.clip(output[0], -1.0, 1.0) * 2.0  
        state, reward, terminated, truncated, _ = env.step([action])
        total_reward += reward
        steps += 1

        if hasattr(env, 'window') and hasattr(env.window, 'window_surface_v2'):
            text = f"Generation: {generation_reached}, Episode: {episode+1}/{max_episodes}, Steps: {steps}"
            position, velocity = state
            text += f"\nPos: {position:.2f}, Vel: {velocity:.2f}, Action: {action}"

        

        if terminated or truncated or steps >= 200:
            print(f"Episode {episode+1} finished after {total_reward} reward")
            total_reward = 0
            episode += 1
            steps = 0
            state, _ = env.reset()
            time.sleep(1)
    
    env.close()

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
    
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), simulate)
    winner = pop.run(pe.evaluate, 300)
    print('\nBest genome:\n{!s}'.format(winner))
    gui(winner, config, generation_reached)
    
if __name__ == '__main__':
    run_neat("pendulum/pendulum_config_ann.txt")