import neat
import numpy as np
import random

class RateIZNN(neat.iznn.IZNN):
    def __init__(self, neurons, inputs, outputs):
        super().__init__(neurons, inputs, outputs)
        self.rate_window = 1  # 1s window
        self.simulation_steps = 200
        self.dt = 0.02
        self.spike_trains = {i: [] for i in outputs}
        self.input_fired = {}  # Track input firing status
        
    def set_inputs(self, inputs):
        """Store normalized inputs [0,1] for probability-based spike generation"""
        if len(inputs) != len(self.inputs):
            raise RuntimeError("Input size mismatch")
        for i, v in zip(self.inputs, inputs):
            # Ensure input is normalized
            self.input_values[i] = max(0.0, min(1.0, v))
            
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
                n.current = n.bias
                
                # Process inputs
                for i, w in n.inputs:
                    if i in self.inputs:
                        # Use pre-determined firing status
                        if self.input_fired[i]:
                            n.current += w
                    else:
                        # Process neuron-to-neuron connections
                        ineuron = self.neurons.get(i)
                        if ineuron is not None:
                            n.current += ineuron.fired * w
                
            for n in self.neurons.values():
                n.advance(self.dt)
                #print(n.v, n.fired, n.current)
                if n.fired > 0:
                    n.spike_count += 1
        
        # Calculate firing rates for outputs
        window_time = self.simulation_steps * self.dt
        output_rates = []
        for i in self.outputs:
            rate = self.neurons[i].spike_count / window_time
            output_rates.append(rate)
            
        return output_rates

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