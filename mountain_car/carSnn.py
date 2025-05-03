import gymnasium as gym
import numpy as np
from rate_iznn import RateIZNN
import multiprocessing
import time


def encode_input(state, min_vals, max_vals, I_min=0, I_max=1):
    norm_state = (state - min_vals) / (max_vals - min_vals)
    I_values = I_min + norm_state * (I_max - I_min)
    return I_values

def decode_output(firing_rates, threshold=0.3):
    # 0: empurrar para esquerda, 1: não empurrar, 2: empurrar para direita
    action = np.argmax(firing_rates)
    
    # Se nenhum neurônio estiver disparando acima do limiar, escolha a ação padrão (não empurrar)
    if max(firing_rates) < threshold:
        return 1
    
    return action

def simulate(genome, config, num_trials=5):
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
            print(output)
            action = decode_output(output[0])
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

def eval_single_genome(genome, config):
    genome.fitness = simulate(genome, config)
    return genome.fitness 

def gui(winner, config, I_min, I_diff, I_background, generation_reached):
    env = gym.make("MountainCar-v0", render_mode="human")
    state, _ = env.reset()
    net = RateIZNN.create(winner, config)
    
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
            net = RateIZNN.create(winner, config)
            time.sleep(1)
    
    env.close()

def run(config_values, config_file, num_Generations=50):  

    config = neat.Config(neat.iznn.IZGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    def create_phenotype(genome):
        return RateIZNN.create(genome, config)

    neat.iznn.IZGenome.create_phenotype = create_phenotype

    config.genome_config.weight_init_mean = config_values["weight_init_mean"]
    config.genome_config.weight_init_stdev = config_values["weight_init_stdev"]
    config.genome_config.weight_max_value = config_values["weight_max_value"]
    config.genome_config.weight_min_value = config_values["weight_min_value"]
    config.genome_config.weight_mutate_power = config_values["weight_mutate_power"]
    config.genome_config.weight_mutate_rate = config_values["weight_mutate_rate"]
    config.genome_config.weight_replace_rate = config_values["weight_replace_rate"]

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
    gui(winner, config, config_values["I_min"], config_values["I_diff"], config_values["background"], generation_reached)


if __name__ == "__main__":
    run({'I_min': -185.20966099570762, 'I_diff': 471, 'background': 50.3531840776152606,'weight_init_mean': 2.0,
    'weight_init_stdev': 2.0,
    'weight_max_value': 30.0,
    'weight_min_value': -20.0,
    'weight_mutate_power': 2.0,
    'weight_mutate_rate': 0.76,
    'weight_replace_rate': 0.2}
        , "mountain_car/mountain_config_snn.txt", 50)