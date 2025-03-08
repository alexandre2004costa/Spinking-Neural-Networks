
import random as ran
import neat
import matplotlib.pyplot as plt
from cartPole import *


class IzhikevichNeuron: 
    def __init__(self, potencial, recovery, threshold, a=0.02, b=0.2, c=-65, d=2):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.v = potencial
        self.u = recovery
        self.vt = threshold # 30 mv
        self.spike_count = 0
        self.sypnatic_input = 0
        self.background_noise = 0.1

    def step(self, dt=0.1): #dt -> derivada do tempo
        spike = False
        if self.v >= self.vt:  # spike reset
            self.v = self.c
            self.u += self.d
            self.spike_count += 1
            spike = True

        dv = 0.04 * self.v ** 2 + 5 * self.v + 140 - self.u + self.sypnatic_input + self.background_noise
        du = self.a * (self.b * self.v - self.u)
        self.sypnatic_input = 0
        #print(f' dv : {dv}, dv * dt {dv * dt}, v : {self.v}')
        self.v += dv * dt
        self.u += du * dt

        return spike

    def receive_spike(self, weight):
        self.v += self.vt * weight

class SNN:
    def __init__(self, number_neuron):
        self.connection_matrix = np.zeros((number_neuron, number_neuron))
        self.neurons = [IzhikevichNeuron(-60, 0.2 * -65, 30, 0.02, 0.2, -50, 2) for _ in range(number_neuron)]
        self.state = np.array([0, 0, 0.05, 0]) # initial cart state
        # Paper config, 1 final node connected to 4 initial ones (inputs)

        #Initial weights
        self.connection_matrix[0][4] = 30
        self.connection_matrix[1][4] = 30
        self.connection_matrix[2][4] = 30
        self.connection_matrix[3][4] = 30

    def update_weights(self, neuron_init, neuron_dest, new_weight):
        self.connection_matrix[neuron_init][neuron_dest] = new_weight
    
    def encode_input(self, state, min_vals, max_vals, I_min=0.1, I_max=5.0):
        norm_state = (state - min_vals) / (max_vals - min_vals)
        I_values = I_min + norm_state * (I_max - I_min)
        return I_values
    
    def generate_spike_train(self, num_spikes, time_window):
        spike_train = np.zeros(time_window)
        spike_times = np.random.choice(time_window, num_spikes, replace=False)
        spike_train[spike_times] = 1
        return spike_train
    
    def propagate_spikes(self, injected_currents, time_window=0.1, dt=0.0001):
        num_steps = int(time_window / dt) # 1000 steps
        for neuron in self.neurons:
            neuron.spike_count = 0  

        spike_input_count = 0
        for t in range(1, num_steps+1):
            for i, neuron in enumerate(self.neurons):
                # Spike input neurons
                if i < 4 and t % int(injected_currents[i] * 10) == 0: # change to probability
                    spike_input_count += 1
                    neuron.sypnatic_input += (neuron.vt - neuron.v)

                # Propagate spikes
                spike = neuron.step()
                if spike: 
                    print(f'Spike {i} fired on {t}')
                    for j in range(len(self.neurons)):
                        if self.connection_matrix[i][j] > 0:
                            self.neurons[j].receive_spike(self.connection_matrix[i][j]) 
        
        print(f'Input spikes : {spike_input_count}')
        return self.neurons[-1].spike_count  # only output spike count
    
    def step(self, action):
        done = False
        reward = 0
        self.state = simulate_cartpole(action, self.state)
        x, _, theta, _ = self.state
        if abs(x) > position_limit or abs(theta) > angle_limit:
            done = True
        reward += (math.cos(theta) + 1)
        return reward, done
       
def build_snn_from_genome(genome):
    snn = SNN(len(genome.nodes))
    for connection in genome.connections.values():
        if connection.enabled:
            snn.update_weights(connection.key[0], connection.key[1], connection.weight)
    return snn

def eval_genome(genome, config):
    #snn = build_snn_from_genome(genome) 
    snn = SNN(5)
    fitness = 0
    done = False
    min_vals = np.array([-position_limit, -2.0, -angle_limit, -2.0])
    max_vals = np.array([ position_limit,  2.0,  angle_limit,  2.0])

    moves = 0
    while not done:
        moves += 1
        injected_currents = snn.encode_input(snn.state, min_vals, max_vals)
        print(injected_currents)
        output_spikes = snn.propagate_spikes(injected_currents)
        print(output_spikes)
        action = 1 if output_spikes > 152 / 2 else 0 # 152 is the number of input spikes , output decode
        reward, done= snn.step(action)
        print((reward, done))
        fitness += reward
        draw_cartpole(snn.state, 0, 0, 0, "oi")

    print(f'Moves : {moves}')
    return fitness

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

def run_neat(config_file):
    config = neat.Config(neat.iznn.IZGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))

    winner = pop.run(eval_genome, 100)  
    print("Best solution found!", winner)

def test_cartpole_snn():
    i = 0
    while(i < 100):
        i+=1
        eval_genome("oi","oi")



if __name__ == '__main__':
    #test_cartpole_snn()
    config_path = "config-feedforward.txt"
    run_neat(config_path)