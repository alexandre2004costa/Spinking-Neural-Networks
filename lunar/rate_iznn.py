import neat
import numpy as np
import random


class RateIZNN(neat.iznn.IZNN):
    def __init__(self, neurons, inputs, outputs, connections):
        super().__init__(neurons, inputs, outputs)
        self.simulation_steps = 100
        self.input_currents = {}  # Store converted input currents
        self.input_fired = {}  # Track input firing status
        self.connections = connections
        self.lastFired = set()
        self.nowFiring = set()
        self.receiving_conn = dict()
        self.neuron_treshold = 30
        
    def set_inputs(self, inputs, I_min=0.0, I_max=10.0):
        """Store normalized inputs [0,1] for probability-based spike generation"""
        if len(inputs) != len(self.inputs):
            raise RuntimeError("Input size mismatch")
        for i, v in zip(self.inputs, inputs):
           self.input_values[i] = v
           self.input_currents[i] = I_min + v * (I_max-I_min)  # Scale input to current range
        
            
    def advance(self, dt):
        for i, n in self.neurons.items():
            n.spike_count = 0
            self.receiving_conn[i] = 0

        for _ in range(self.simulation_steps):
            print(self.connections)
            self.nowFiring.clear()
            self.input_fired.clear()

            for i in self.inputs:
                self.input_fired[i] = random.random() < self.input_values[i]
                if self.input_fired[i]:
                    self.nowFiring.add(i)

            for i, out in self.connections.items():
                if i in self.lastFired:
                    if i in self.inputs:
                        for o, w in out:
                            self.receiving_conn[o] += w * self.input_currents[i]
                    else:
                        for o, w in out:
                            self.receiving_conn[o] += w * 10

            for i, n in self.neurons.items():
                if i in self.lastFired:
                    Sout = 1
                else:
                    Sout = 0

                if i in self.outputs:
                    n.current = n.current * 0.3 + self.receiving_conn[o] - Sout * self.neuron_treshold
                else:
                    n.current = n.current * 0.5 + self.receiving_conn[o] - Sout * self.neuron_treshold
                
                n.advance(dt)
                if n.fired > 0:
                    self.nowFiring.add(i)
                    n.spike_count += 1

            self.lastFired = self.nowFiring

        return [self.neurons[i].spike_count  for i in self.outputs]


    @staticmethod
    def create(genome, config):
        neurons = {}
        inputs = []
        outputs = []
        connections = {}
        
        # First add input nodes (they're not in genome.nodes)
        for input_id in config.genome_config.input_keys:
            inputs.append(input_id)
        
        # Then add output and hidden nodes from genome
        for key, ng in genome.nodes.items():
            neurons[key] = neat.iznn.IZNeuron(ng.bias, ng.a, ng.b, ng.c, ng.d, [])
            if key in config.genome_config.output_keys:
                outputs.append(key)
        
        # Add connections
        for key, cg in genome.connections.items():
            if cg.enabled:
                i, o = key
                if i in connections.keys():
                    connections[i] += (o, cg.weight)
                else:
                    connections[i] = (o, cg.weight)                    
        
        return RateIZNN(neurons, inputs, outputs, connections)