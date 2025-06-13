import gymnasium as gym
import neat
import numpy as np
import time
import multiprocessing
import time
from stats import RLStatsCollector
from mountain_car.customIzGenome import CustomIZGenome
from rate_iznn import RateIZNN



def encode_input(state, min_vals, max_vals, I_min=0, I_max=1):
    norm_state = (state - min_vals) / (max_vals - min_vals)
    I_values = I_min + norm_state * (I_max - I_min)
    return I_values

def decode_output(firing_rates):
    action = np.argmax(firing_rates)
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
    start_time = time.time()
    config = neat.Config(CustomIZGenome, neat.DefaultReproduction,
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
    elapsed_time = time.time() - start_time

    print(winner)
    print(f"STEPS : {winner.simulation_steps}")
    print(f"I MAX : {winner.input_scaling}")
    print(f"I MIN : {winner.input_min}")
    print(f"BACKGROUND : {winner.background}")
    print(f"Total time : {elapsed_time:.2f} sec")
    gui(winner, config, generation_reached)


def run_experiment(config_file, num_Generations=50):
    config = neat.Config(CustomIZGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    fitness_threshold = config.fitness_threshold if hasattr(config, "fitness_threshold") else 100000
    stats_collector = RLStatsCollector(fitness_threshold=fitness_threshold)
    stats_collector.start_experiment()

    def create_phenotype(genome):
        return RateIZNN.create(genome, config)
    neat.iznn.IZGenome.create_phenotype = create_phenotype

    pop = neat.population.Population(config)
    generation_reached = 0

    class GenerationReporter(neat.reporting.BaseReporter):
        def start_generation(self, generation):
            nonlocal generation_reached
            generation_reached = generation
            self.generation_start_time = time.time()

        def post_evaluate(self, config, population, species, best_genome):
            gen_time = time.time() - self.generation_start_time
            stats_collector.record_generation(best_genome.fitness, gen_time)

    pop.add_reporter(GenerationReporter())
    pop.add_reporter(neat.StdOutReporter(False))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_single_genome)
    winner = pop.run(pe.evaluate, num_Generations)
    stats_collector.end_experiment()

    # Average decision time and total reward
    env = gym.make("MountainCar-v0", render_mode=None)
    state, _ = env.reset()
    net = RateIZNN.create(winner, config)
    action_times = []
    total_reward = 0
    steps = 0
    done = False

    while not done and steps < 1000:
        input_values = encode_input(state, env.observation_space.low, env.observation_space.high)
        net.set_inputs(input_values)
        t0 = time.time()
        output = net.advance(0.02)
        action = decode_output(output)
        t1 = time.time()
        action_times.append(t1 - t0)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    env.close()

    return {
        "learning_time": stats_collector.learning_time(),
        "success": 1 if stats_collector.success_generation else 0,
        "mean_decision_time": np.mean(action_times) if action_times else 0,
        "best_fitness": max(stats_collector.fitness_history) if stats_collector.fitness_history else 0,
        "mean_fitness": np.mean(stats_collector.fitness_history) if stats_collector.fitness_history else 0,
        "hidden_nodes": len(winner.nodes) - len(config.genome_config.output_keys),
        "total_generations": len(stats_collector.fitness_history),
        "winner_steps": getattr(winner, "simulation_steps", None),
        "winner_input_scaling": getattr(winner, "input_scaling", None),
        "winner_input_min": getattr(winner, "input_min", None),
        "winner_background": getattr(winner, "background", None),
        "final_reward": total_reward
    }

if __name__ == "__main__":
    print("Running Mountain Car SNN experiment...")
    #run("mountain_car/mountain_config_snn.txt", 50)