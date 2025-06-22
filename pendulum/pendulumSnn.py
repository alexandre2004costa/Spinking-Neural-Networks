import gymnasium as gym
import numpy as np
from rate_iznn_cont import RateIZNN
import multiprocessing
import time
import neat
from pendulum.customIzGenome import CustomIZGenome
from stats import RLStatsCollector


def encode_input(state, min_vals, max_vals, I_min=0, I_max=1):
    norm_state = (state - min_vals) / (max_vals - min_vals)
    I_values = I_min + norm_state * (I_max - I_min)
    return I_values


def compute_force(weighted_sum, sigma=1.0):
    try:
        Fn = 1.0 / (1.0 + np.exp(-sigma * (weighted_sum)))
        Ft = 2 * (2 * Fn - 1)
        #print(f"WS : {weighted_sum} Fn: {Fn}, Ft: {Ft}")
        return Ft
    except OverflowError:
        return 2.0


def simulate(genome, config, num_trials=5):
    trials_reward = []
    
    for _ in range(num_trials):
        net = RateIZNN.create(genome, config)   
        env = gym.make("Pendulum-v1", render_mode=None)
        state, _ = env.reset()
        
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            input_values = encode_input(state, env.observation_space.low, env.observation_space.high)
            net.set_inputs(input_values)
            output = net.advance(0.01)
            action = compute_force(output, genome.sigma)
            #print(f"Outp : {output},  ACTION : {action}, {np.array([action])}")
            state, reward, terminated, truncated, _ = env.step(np.array([action], dtype=np.float32))  
            
            total_reward += reward
            steps += 1
            done = terminated or truncated or steps >= 200

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
    env = gym.make("Pendulum-v1", render_mode="human")
    state, _ = env.reset()
    net = RateIZNN.create(winner, config)
    
    episode = 0
    steps = 0
    total_reward = 0
    
    while episode < 10:
        input_values = encode_input(state, env.observation_space.low, env.observation_space.high)
        net.set_inputs(input_values)
        output = net.advance(0.01)
        action = compute_force(output, winner.sigma)
        #print(f"Outp : {output},  ACTION : {action}, {np.array([action])}")
        state, reward, terminated, truncated, _ = env.step(np.array([action], dtype=np.float32))  
        
        total_reward += reward
        steps += 1

        if terminated or truncated or steps >= 200:
            print(f"Episode {episode+1} finished after {steps} steps with total reward: {total_reward}")
            episode += 1
            steps = 0
            total_reward = 0
            state, _ = env.reset()
            net = RateIZNN.create(winner, config)
            time.sleep(1)

    
    env.close()

def run(config_file, num_Generations=50):  

    config = neat.Config(CustomIZGenome , neat.DefaultReproduction,
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
    print(f"STEPS : {winner.simulation_steps}")
    print(f"I MAX : {winner.input_scaling}")
    print(f"I MIN : {winner.input_min}")
    print(f"BACKGROUND : {winner.background}")
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
    success = 1 if stats_collector.success_generation else 0
    if success:
        # Average decision time and total reward
        env = gym.make("Pendulum-v1", render_mode=None)
        state, _ = env.reset()
        net = RateIZNN.create(winner, config)
        action_times = []
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < 200:
            input_values = encode_input(state, env.observation_space.low, env.observation_space.high)
            net.set_inputs(input_values)
            t0 = time.time()
            output = net.advance(0.01)
            action = compute_force(output, winner.sigma)
            t1 = time.time()
            action_times.append(t1 - t0)
            state, reward, terminated, truncated, _ = env.step(np.array([action], dtype=np.float32))
            total_reward += reward
            steps += 1
            done = terminated or truncated or steps >= 200

        env.close()

        return {
            "learning_time": stats_collector.learning_time(),
            "success": 1,
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
    
    return {
        "learning_time": None,
        "success": 0,
        "mean_decision_time": None,
        "best_fitness": None,
        "mean_fitness": None,
        "hidden_nodes": None,
        "total_generations": None,
        "winner_steps": None,
        "winner_input_scaling": None,
        "winner_input_min": None,
        "winner_background": None,
        "final_reward": None
    } 

if __name__ == "__main__":
    print("Running Pendulum SNN experiment...")
    #run("pendulum/pendulum_config_snn.txt", 100)