import neat
import numpy as np
import random

class RateIZNN(neat.iznn.IZNN):
    def __init__(self, neurons, inputs, outputs):
        super().__init__(neurons, inputs, outputs)
        self.rate_window = 1  # 1s window
        self.simulation_steps = 400
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
        # Reset spike counts
        for n in self.neurons.values():
            n.spike_count = 0
        
        # Clear output spike trains
        for o in self.outputs:
            self.spike_trains[o] = []
            
        # Simulate for multiple steps
        for _ in range(self.simulation_steps):
            # First determine which inputs fire this step
            self.input_fired.clear()
            for i in self.inputs:
                # Generate input spike probabilistically
                self.input_fired[i] = random.random() < self.input_values[i]
            
            
            # Then update neurons using pre-determined input spikes
            for n in self.neurons.values():
                n.current = n.bias + random.gauss(0, 0.1) # Add background noise
                
                # Process inputs
                for i, w in n.inputs:
                    if i in self.inputs:
                        # Use pre-determined firing status
                        if self.input_fired[i]:
                            #n.current += self.input_currents[i] * w
                            n.current += w
                    else:
                        # Process neuron-to-neuron connections
                        ineuron = self.neurons.get(i)
                        if ineuron is not None:
                            n.current += ineuron.fired * w
                
                n.advance(self.dt)
                if n.fired > 0:
                    n.spike_count += 1
            #print("Neurons fired:", self.input_fired)
            #print("neuron inputs:", self.input_values)
        
        # Calculate firing rates for outputs
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