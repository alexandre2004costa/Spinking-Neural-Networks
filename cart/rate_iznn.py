import neat
import numpy as np
import random


class RateIZNN(neat.iznn.IZNN):
    def __init__(self, neurons, inputs, outputs):
        super().__init__(neurons, inputs, outputs)
        self.simulation_steps = 300
        self.dt = 0.02
        self.spike_trains = {i: [] for i in outputs}
        self.input_currents = {}  # Store converted input currents
        self.input_fired = {}  # Track input firing status
        
    def set_inputs(self, inputs, I_min=20.0, I_max=100.0):
        """Store normalized inputs [0,1] for probability-based spike generation"""
        if len(inputs) != len(self.inputs):
            raise RuntimeError("Input size mismatch")
        for i, v in zip(self.inputs, inputs):
           self.input_values[i] = v
           self.input_currents[i] = I_min + v * I_max  # Scale input to current range
            
    def advance(self, dt):
        for n in self.neurons.values():
            n.spike_count = 0
        for o in self.outputs:
            self.spike_trains[o] = []

        for _ in range(self.simulation_steps):
            self.input_fired.clear()
            for i in self.inputs:
                self.input_fired[i] = random.random() < self.input_values[i]

            # --- Fase 1: Propagação dos inputs para os hidden ---
            for i, n in self.neurons.items():
                if i in self.outputs:
                    continue  # só tratamos hidden nesta fase

                n.current = n.bias # background

                for j, w in n.inputs:
                    if j in self.inputs:
                        if self.input_fired[j]:
                            n.current += w
                            n.current += w
                    else:
                        ineuron = self.neurons.get(j)
                        if ineuron is not None:
                            n.current += ineuron.fired * w

            # Update hidden neurons
            for i, n in self.neurons.items():
                if i in self.outputs:
                    continue
                #print("Neuron", i)
                #print(n.current, n.v, n.bias)
                n.advance(self.dt)
                #print(n.v, n.u, n.fired, n.spike_count, n.current)
                if n.fired > 0:
                    n.spike_count += 1

            # --- Fase 2: Propagação dos hidden para os output ---
            for i in self.outputs:
                n = self.neurons[i]
                n.current = n.bias # background

                for j, w in n.inputs:
                    if j in self.inputs:
                        if self.input_fired[j]:
                            n.current += w
                    else:
                        ineuron = self.neurons.get(j)
                        if ineuron is not None:
                            n.current += ineuron.fired * w

            # Update output neurons
            for i in self.outputs:
                n = self.neurons[i]
                #print(n.current)
                n.advance(self.dt)
                #print(n.v, n.u, n.fired, n.spike_count, n.current)
                if n.fired > 0:
                    n.spike_count += 1

       # print("Spike counts:", [self.neurons[i].spike_count for i, j in self.neurons.items()])
        window_time = self.simulation_steps * self.dt
        return [self.neurons[i].spike_count / window_time for i in self.outputs]


    @staticmethod
    def create(genome, config):
        neurons = {}
        inputs = []
        outputs = []
        
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
                neurons[o].inputs.append((i, cg.weight))
        
        return RateIZNN(neurons, inputs, outputs)