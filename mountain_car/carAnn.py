import gymnasium as gym
import numpy as np
import neat
import time
import multiprocessing
from stats import RLStatsCollector

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

def run(config_file, num_Generations=50):
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
    winner = pop.run(pe.evaluate, num_Generations)
    elapsed_time = time.time() - start_time
    print('\nBest genome:\n{!s}'.format(winner))
    print('\nElapsed time: {:.2f} seconds'.format(elapsed_time))
    gui(winner, config, generation_reached)
    
def run_experiment(config_file, num_Generations=50):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                        neat.DefaultSpeciesSet, neat.DefaultStagnation, 
                        config_file)
    fitness_threshold = config.fitness_threshold if hasattr(config, "fitness_threshold") else 100000
    stats_collector = RLStatsCollector(fitness_threshold=fitness_threshold)
    stats_collector.start_experiment()

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(False))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    generation = 0
    class GenerationReporter(neat.reporting.BaseReporter):
        def start_generation(self, gen):
            nonlocal generation
            generation = gen
            self.generation_start_time = time.time()
        def post_evaluate(self, config, population, species, best_genome):
            gen_time = time.time() - self.generation_start_time
            stats_collector.record_generation(best_genome.fitness, gen_time)

    population.add_reporter(GenerationReporter())
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), simulate)
    winner = population.run(pe.evaluate, num_Generations)

    stats_collector.end_experiment()
    elapsed_time = stats_collector.learning_time()

     # Evaluate winner average time to take an action
    env = gym.make("MountainCar-v0", render_mode=None)
    state, _ = env.reset()
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    action_times = []
    total_reward = 0
    steps = 0
    done = False

    success = 1 if stats_collector.success_generation else 0
    if success:
        while not done and steps < 1000:
            normalized_state = (state - env.observation_space.low) / (
                env.observation_space.high - env.observation_space.low
            )
            t0 = time.time()
            output = net.activate(normalized_state)
            action = np.argmax(output)
            t1 = time.time()
            action_times.append(t1 - t0)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        env.close()

        num_hidden = config.genome_config.num_hidden if hasattr(config.genome_config, "num_hidden") else None
        best_fitness = max(stats_collector.fitness_history) if stats_collector.fitness_history else 0
        mean_fitness = np.mean(stats_collector.fitness_history) if stats_collector.fitness_history else 0
        success = 1 if stats_collector.success_generation else 0
        mean_decision_time = np.mean(action_times) if action_times else 0

        return {
            "learning_time": elapsed_time,
            "success": success,
            "mean_decision_time": mean_decision_time,
            "best_fitness": best_fitness,
            "mean_fitness": mean_fitness,
            "hidden_nodes": num_hidden,
            "total_generations": generation+1,
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
        "final_reward": total_reward
    }
if __name__ == '__main__':
    print("running Ann experiment")
    #run("mountain_car/mountain_config_ann.txt", 100)