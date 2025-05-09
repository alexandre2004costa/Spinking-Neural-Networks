import gymnasium as gym
import numpy as np
from rate_iznn import RateIZNN
import multiprocessing
import time
import neat


def encode_input(state, min_vals, max_vals, I_min=0, I_max=1):
    norm_state = (state - min_vals) / (max_vals - min_vals)
    I_values = I_min + norm_state * (I_max - I_min)
    return I_values

def decode_output(firing_rates):
    action = np.argmax(firing_rates)
    return action

def simulate(genome, config, num_trials=10):
    trials_reward = []
    
    for _ in range(num_trials):
        net = RateIZNN.create(genome, config)   
        env = gym.make("MountainCar-v0", render_mode=None)
        state, _ = env.reset()
        
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            input_values = encode_input(state, env.observation_space.low, env.observation_space.high)
            net.set_inputs(input_values)

            output = net.advance(0.02)
            #print(output)
            action = decode_output(output)
            state, reward, terminated, truncated, _ = env.step(action)  
            
            total_reward += reward
            steps += 1
            done = terminated or truncated or steps >= 1000

            #net.reset() # Reset the network for the next iteration
        
        env.close()
        trials_reward.append(float(total_reward))
    
    avg_reward = sum(trials_reward) / num_trials
    if np.isnan(avg_reward) or np.isinf(avg_reward):
        return 0.0
    
    return avg_reward

def eval_single_genome(genome, config):
    genome.fitness = simulate(genome, config)
    return genome.fitness 

def gui(winner, config, generation_reached):
    env = gym.make("MountainCar-v0", render_mode="human")
    state, _ = env.reset()
    net = RateIZNN.create(winner, config)
    
    episode = 0
    steps = 0
    
    while episode < 10:
        input_values = encode_input(state, env.observation_space.low, env.observation_space.high)
        net.set_inputs(input_values)
        output = net.advance(0.02)
        action = decode_output(output)
        state, reward, terminated, truncated, _ = env.step(action)  
        steps += 1
        
        if hasattr(env, 'window') and hasattr(env.window, 'window_surface_v2'):
            text = f"Generation: {generation_reached}, Episode: {episode+1}/{10}, Steps: {steps}"
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
            net = RateIZNN.create(winner, config)
            time.sleep(1)

    
    env.close()

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
    run("mountain_car/mountain_config_snn.txt", 50)