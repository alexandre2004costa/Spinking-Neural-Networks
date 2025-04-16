import time
import numpy as np
import matplotlib.pyplot as plt
from carAnn import run_neat as run_ann
from carSnn import config_values, run as run_snn, simulate
import pandas as pd
from datetime import datetime

def benchmark(num_trials=10):
    ann_stats = []
    snn_stats = []
    
    print(f"Starting benchmark with {num_trials} trials for each implementation...")
    
    # Run ANN trials
    print("\nRunning ANN trials:")
    for i in range(num_trials):
        print(f"\nTrial {i+1}/{num_trials}")
        start_time = time.time()
        
        class GenerationCounter:
            def __init__(self):
                self.last_gen = 0
            def get_last_gen(self):
                return self.last_gen
        
        gen_counter = GenerationCounter()
        
        # Run ANN without GUI
        run_ann("car/mountain_config_ann.txt")
        end_time = time.time()
        
        ann_stats.append({
            'generations': gen_counter.get_last_gen(),
            'time': end_time - start_time
        })
    
    # Run SNN trials
    print("\nRunning SNN trials:")
    for i in range(num_trials):
        print(f"\nTrial {i+1}/{num_trials}")
        start_time = time.time()
        
        class GenerationCounter:
            def __init__(self):
                self.last_gen = 0
            def get_last_gen(self):
                return self.last_gen
        
        gen_counter = GenerationCounter()
        
        # Run SNN without GUI
        run_snn(config_values, lambda *args: simulate(*args), 
                "car/mountain_config_snn.txt", None, 100)
        end_time = time.time()
        
        snn_stats.append({
            'generations': gen_counter.get_last_gen(),
            'time': end_time - start_time
        })
    
    # Calculate statistics
    ann_gens = [stat['generations'] for stat in ann_stats]
    ann_times = [stat['time'] for stat in ann_stats]
    snn_gens = [stat['generations'] for stat in snn_stats]
    snn_times = [stat['time'] for stat in snn_stats]
    
    stats = {
        'ANN': {
            'generations': {
                'mean': np.mean(ann_gens),
                'std': np.std(ann_gens),
                'min': np.min(ann_gens),
                'max': np.max(ann_gens)
            },
            'time': {
                'mean': np.mean(ann_times),
                'std': np.std(ann_times),
                'min': np.min(ann_times),
                'max': np.max(ann_times)
            }
        },
        'SNN': {
            'generations': {
                'mean': np.mean(snn_gens),
                'std': np.std(snn_gens),
                'min': np.min(snn_gens),
                'max': np.max(snn_gens)
            },
            'time': {
                'mean': np.mean(snn_times),
                'std': np.std(snn_times),
                'min': np.min(snn_times),
                'max': np.max(snn_times)
            }
        }
    }
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_df = pd.DataFrame({
        'ANN_generations': ann_gens,
        'ANN_time': ann_times,
        'SNN_generations': snn_gens,
        'SNN_time': snn_times
    })
    results_df.to_csv(f'benchmark_results_{timestamp}.csv')
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Generations comparison
    ax1.boxplot([ann_gens, snn_gens], labels=['ANN', 'SNN'])
    ax1.set_title('Generations to Solution')
    ax1.set_ylabel('Number of Generations')
    
    # Time comparison
    ax2.boxplot([ann_times, snn_times], labels=['ANN', 'SNN'])
    ax2.set_title('Time to Solution')
    ax2.set_ylabel('Time (seconds)')
    
    plt.savefig(f'benchmark_plots_{timestamp}.png')
    plt.close()
    
    # Print summary
    print("\nBenchmark Results:")
    print("\nArtificial Neural Network (ANN):")
    print(f"Generations: {stats['ANN']['generations']['mean']:.2f} ± {stats['ANN']['generations']['std']:.2f}")
    print(f"Time (s): {stats['ANN']['time']['mean']:.2f} ± {stats['ANN']['time']['std']:.2f}")
    
    print("\nSpiking Neural Network (SNN):")
    print(f"Generations: {stats['SNN']['generations']['mean']:.2f} ± {stats['SNN']['generations']['std']:.2f}")
    print(f"Time (s): {stats['SNN']['time']['mean']:.2f} ± {stats['SNN']['time']['std']:.2f}")
    
    return stats

if __name__ == '__main__':
    benchmark(num_trials=10)