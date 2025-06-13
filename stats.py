import time
import numpy as np
import csv

class RLStatsCollector:
    def __init__(self, fitness_threshold):
        self.fitness_threshold = fitness_threshold
        self.start_time = None
        self.end_time = None
        self.generation_times = []
        self.fitness_history = []
        self.success_generation = None
        self.action_times = []

    def start_experiment(self):
        self.start_time = time.time()

    def end_experiment(self):
        self.end_time = time.time()

    def record_generation(self, fitness, gen_time):
        self.fitness_history.append(fitness)
        self.generation_times.append(gen_time)
        if self.success_generation is None and fitness >= self.fitness_threshold:
            self.success_generation = len(self.fitness_history)

    def record_action_time(self, action_time):
        self.action_times.append(action_time)

    def learning_time(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    def success_rate(self, total_runs):
        return 1.0 if self.success_generation is not None else 0.0

    def mean_decision_time(self):
        return np.mean(self.action_times) if self.action_times else None

    def report(self, total_runs=1):
        print("=== RL Experiment Statistics ===")
        print(f"Total learning time: {self.learning_time():.2f} seconds")
        print(f"Success (reached fitness threshold): {'Yes' if self.success_generation else 'No'}")
        if self.success_generation:
            print(f"Generation of first success: {self.success_generation}")
        print(f"Success rate: {self.success_rate(total_runs)*100:.1f}%")
        print(f"Mean decision time per action: {self.mean_decision_time():.6f} seconds")
        print(f"Best fitness achieved: {max(self.fitness_history) if self.fitness_history else None}")
        print(f"Mean fitness: {np.mean(self.fitness_history) if self.fitness_history else None}")
        print(f"Total generations: {len(self.fitness_history)}")
        print("================================")

def run_stats(Runs, NumMaxGenerations, experiment_config, result_file, run_experiment):
    results = []
    for i in range(Runs):
        print(f"Run {i+1}/{Runs}")
        stats = run_experiment(experiment_config, NumMaxGenerations)
        results.append(stats)

    # Calculate max, min, and mean for each statistic
    keys = results[0].keys()
    summary = {}
    for key in keys:
        values = [r[key] for r in results]
        summary[key+"_mean"] = np.mean(values)
        summary[key+"_min"] = np.min(values)
        summary[key+"_max"] = np.max(values)

    # Write results to CSV
    FileName = result_file + str(Runs) + ".csv"
    with open("results/" + FileName, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["stat", "mean", "min", "max"])
        for key in keys:
            mean = summary[key+"_mean"]
            minv = summary[key+"_min"]
            maxv = summary[key+"_max"]
            def fmt(val):
                if val == 0:
                    return "0.00"
                if isinstance(val, float):
                    if abs(val) < 1e-3 or abs(val) > 1e6:
                        return f"{val:.2e}"
                    else:
                        return f"{val:.2f}"
                return val
            writer.writerow([key, fmt(mean), fmt(minv), fmt(maxv)])


    print("Results saved to ", FileName)