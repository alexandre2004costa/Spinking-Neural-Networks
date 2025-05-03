import neat
import numpy as np
import random

class RateIZNN(neat.iznn.IZNN):
    def __init__(self, neurons, inputs, outputs, connections):
        super().__init__(neurons, inputs, outputs)
        self.simulation_steps = 100
        self.spike_trains = {i: [] for i in outputs}
        self.input_currents = {} 
        self.input_fired = {} 
        self.connections = connections
        #self.monitor = SpikeMonitor(self)
        self.nowFiring = set()
        self.receiving_conn = dict()
        self.input_firing_schedule = {}
        
    def set_inputs(self, inputs, I_min=0.0, I_max=100.0):
        if len(inputs) != len(self.inputs):
            raise RuntimeError("Input size mismatch")
        for i, v in zip(self.inputs, inputs):
           self.input_values[i] = v
           self.input_currents[i] = I_min + v * I_max  

        # Defining the schedule of input spikes
        self.input_firing_schedule = {}
        for i in self.inputs:
            num_fires = int(self.input_values[i] * self.simulation_steps)
            if num_fires > 0:
                firing_steps = set(np.linspace(0, self.simulation_steps-1, num_fires, dtype=int))
            else:
                firing_steps = set()
            self.input_firing_schedule[i] = firing_steps
            
    def advance(self, dt):
        for i, n in self.neurons.items():
            n.spike_count = 0
            self.receiving_conn[i] = 0

        for t in range(self.simulation_steps):
            self.nowFiring.clear()
            self.input_fired.clear()

            # Input spiking
            for i in self.inputs:
                self.input_fired[i] = t in self.input_firing_schedule[i]
                if self.input_fired[i]:
                    self.nowFiring.add(i)

            # Propagate spikes
            for i in self.nowFiring:
                if i in self.connections:
                    for o, w in self.connections[i]:
                        if i in self.inputs:
                            self.receiving_conn[o] += w * self.input_currents[i]
                        else:
                            self.receiving_conn[o] += w * 10 + 100

            # Process current in neurons
            for i, n in self.neurons.items():
                n.current = n.bias + self.receiving_conn[i]
                n.advance(dt)
                if n.fired > 0:
                    self.nowFiring.add(i)
                    n.spike_count += 1

            # Clean receivers
            for i in self.receiving_conn:
                self.receiving_conn[i] = 0

        
        return [self.neurons[i].spike_count for i in self.outputs]

    @staticmethod
    def create(genome, config):
        neurons = {}
        inputs = []
        outputs = []
        connections = {}
        
        # Add input nodes
        for input_id in config.genome_config.input_keys:
            inputs.append(input_id)
        
        # Add output and hidden nodes 
        for key, ng in genome.nodes.items():
            neurons[key] = neat.iznn.IZNeuron(ng.bias, ng.a, ng.b, ng.c, ng.d, [])
            if key in config.genome_config.output_keys:
                outputs.append(key)
        
        # Add connections
        for key, cg in genome.connections.items():
            if cg.enabled:
                i, o = key
                neurons[o].inputs.append((i, cg.weight))
                if i in connections.keys():
                    connections[i].append((o, cg.weight))
                else:
                    connections[i] = [(o, cg.weight)]                    
        
        return RateIZNN(neurons, inputs, outputs, connections)