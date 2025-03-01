import numpy as np
import random as ran
import neat
import matplotlib.pyplot as plt

class IzhikevichNeuron: 
    def __init__(self, potencial, recovery, threshold, a=0.02, b=0.2, c=-65, d=2):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.v = potencial
        self.u = recovery
        self.vt = threshold # 30 mv

    def step(self, injected_sypnatic, dt=1.0): #dt -> derivada do tempo, assumir 1 seg 

        if self.v >= self.vt:  # spike reset
            self.v = self.c
            self.u += self.d

        dv = 0.04 * self.v ** 2 + 5 * self.v + 140 - self.u + injected_sypnatic
        du = self.a * (self.b * self.v - self.u)

        self.v += dv * dt
        self.u += du * dt

    def receive_spike(self, weight):
        self.potencial += self.threshold * weight

    def background_noise(self, noise = 0.1):
        self.potencial += noise

    def spike(self):
        return self.v >= self.vt

class SNN:
    def __init__(self, number_neuron):
        self.connection_matrix = np.zeros((number_neuron, number_neuron))
        self.neurons = [IzhikevichNeuron(65, 0.2 * -65, 30, 0.02, 0.2, -50, 2) for _ in range(number_neuron)]
        
        # Paper config, 1 final node connected to 4 initial ones (inputs)
        self.connection_matrix[0][4] = ran.random()
        self.connection_matrix[1][4] = ran.random()
        self.connection_matrix[2][4] = ran.random()
        self.connection_matrix[3][4] = ran.random()

    def update_weights(self, neuron_init, neuron_dest, new_weight):
        self.connection_matrix[neuron_init][neuron_dest] = new_weight

    def propagate_spikes(self):
        for i, neuron in enumerate(self.neurons):
            for j in range(len(self.neurons)):
                if self.connection_matrix[i][j] > 0:  # In case of connection
                    self.neurons[j].receive_spike(self.connection_matrix[i][j])
            
            neuron.step()
            neuron.background_noise()
        
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    snn = SNN(5)  # 5 neurônios na SNN
    
    for i in range(100):  # Simulação
        inputs = np.random.rand(4)  # Inputs do CartPole
        weights = net.activate(inputs)  # NEAT gera os pesos sinápticos
        snn.update_weights(0, 4, weights[0])  # Atualiza conexão
        snn.propagate_spikes()
    
    return some_fitness_value  # Retorna a fitness

if __name__ == '__main__':

    config_path = "config-feedforward"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
    pop = neat.Population(config)
    pop.run(eval_genome, 100)