import gymnasium as gym
import numpy as np
import neat
import time
import multiprocessing


def decode_output(firing_rates):
    action = np.argmax(firing_rates)
    return action
    
def simulate(genome, config, num_trials=5):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    trials_reward = []
    
    for _ in range(num_trials):
        env = gym.make("MountainCar-v0", render_mode=None)
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            normalized_state = (state - env.observation_space.low) / (
                env.observation_space.high - env.observation_space.low
            )
            output = net.activate(normalized_state)
            action = decode_output(output)
            
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
    env = gym.make("MountainCar-v0", render_mode="human")
    state, _ = env.reset()
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    episode = 0
    steps = 0
    max_episodes = 10
    
    while episode < max_episodes:
        normalized_state = (state - env.observation_space.low) / (
            env.observation_space.high - env.observation_space.low
        )
        
        output = net.activate(normalized_state)
        action = decode_output(output)
        
        state, reward, terminated, truncated, _ = env.step(action)
        steps += 1

        if hasattr(env, 'window') and hasattr(env.window, 'window_surface_v2'):
            text = f"Generation: {generation_reached}, Episode: {episode+1}/{max_episodes}, Steps: {steps}"
            position, velocity = state
            text += f"\nPos: {position:.2f}, Vel: {velocity:.2f}, Action: {action}"

        

        if terminated or truncated or steps >= 1000:
            if state[0] >= 0.5:
                print(f"Episódio {episode+1}: Sucesso! Alcançou o topo em {steps} passos.")
            else:
                print(f"Episódio {episode+1}: Falha. Posição máxima: {state[0]:.2f}")
            
            episode += 1
            steps = 0
            state, _ = env.reset()
            time.sleep(1)
    
    env.close()

def run_neat(config_file):
    start_time = time.time()
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
    winner = pop.run(pe.evaluate, 100)
    elapsed_time = time.time() - start_time
    print('\nBest genome:\n{!s}'.format(winner))
    print('\nElapsed time: {:.2f} seconds'.format(elapsed_time))
    #gui(winner, config, generation_reached)
    
if __name__ == '__main__':
    run_neat("mountain_car/mountain_config_ann.txt")