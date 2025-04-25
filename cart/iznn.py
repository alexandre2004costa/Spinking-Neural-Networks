"""
This module implements a spiking neural network.
Neurons are based on the model described by:

Izhikevich, E. M.
Simple Model of Spiking Neurons
IEEE TRANSACTIONS ON NEURAL NETWORKS, VOL. 14, NO. 6, NOVEMBER 2003

http://www.izhikevich.org/publications/spikes.pdf
"""

from neat.attributes import FloatAttribute
from neat.genes import BaseGene, DefaultConnectionGene
from neat.genome import DefaultGenomeConfig, DefaultGenome
from neat.graphs import required_for_output
import random
import numpy as np

# a, b, c, d are the parameters of the Izhikevich model.
# a: the time scale of the recovery variable
# b: the sensitivity of the recovery variable
# c: the after-spike reset value of the membrane potential
# d: after-spike reset of the recovery variable
# The following parameter sets produce some known spiking behaviors:
# pylint: disable=bad-whitespace
REGULAR_SPIKING_PARAMS        = {'a': 0.02, 'b': 0.20, 'c': -65.0, 'd': 8.00}
INTRINSICALLY_BURSTING_PARAMS = {'a': 0.02, 'b': 0.20, 'c': -55.0, 'd': 4.00}
CHATTERING_PARAMS             = {'a': 0.02, 'b': 0.20, 'c': -50.0, 'd': 2.00}
FAST_SPIKING_PARAMS           = {'a': 0.10, 'b': 0.20, 'c': -65.0, 'd': 2.00}
THALAMO_CORTICAL_PARAMS       = {'a': 0.02, 'b': 0.25, 'c': -65.0, 'd': 0.05}
RESONATOR_PARAMS              = {'a': 0.10, 'b': 0.25, 'c': -65.0, 'd': 2.00}
LOW_THRESHOLD_SPIKING_PARAMS  = {'a': 0.02, 'b': 0.25, 'c': -65.0, 'd': 2.00}


# TODO: Add mechanisms analogous to axon & dendrite propagation delay.


class IZNodeGene(BaseGene):
    """Contains attributes for the iznn node genes and determines genomic distances."""

    _gene_attributes = [FloatAttribute('bias'),
                        FloatAttribute('a'),
                        FloatAttribute('b'),
                        FloatAttribute('c'),
                        FloatAttribute('d')]

    def distance(self, other, config):
        s = abs(self.a - other.a) + abs(self.b - other.b) \
            + abs(self.c - other.c) + abs(self.d - other.d)
        return s * config.compatibility_weight_coefficient


class IZGenome(DefaultGenome):
    @classmethod
    def parse_config(cls, param_dict):
        # Convert numeric parameters from strings to integers
        num_inputs = int(param_dict['num_inputs'])
        num_outputs = int(param_dict['num_outputs'])
        
        # Set input and output keys
        param_dict['input_keys'] = [-i-1 for i in range(num_inputs)]
        param_dict['output_keys'] = [i for i in range(num_outputs)]
        
        # Set gene types
        param_dict['node_gene_type'] = IZNodeGene
        param_dict['connection_gene_type'] = DefaultConnectionGene
        
        # Convert other numeric parameters
        numeric_params = ['a_init_mean', 'b_init_mean', 'c_init_mean', 'd_init_mean',
                         'weight_init_mean', 'weight_init_stdev', 'weight_max_value',
                         'weight_min_value', 'weight_mutate_power', 'weight_mutate_rate',
                         'weight_replace_rate']
                         
        for param in numeric_params:
            if param in param_dict:
                param_dict[param] = float(param_dict[param])
        
        return DefaultGenomeConfig(param_dict)

    def configure_new(self, config):
        # Initialize node genes
        self.nodes = {}
        
        # Create input nodes (negative IDs)
        for node_id in config.input_keys:
            self.nodes[node_id] = self.create_node(config, node_id)
            
        # Create output nodes (positive IDs)
        for node_id in config.output_keys:
            self.nodes[node_id] = self.create_node(config, node_id)
            
        # Create initial connections
        self.connections = {}
        if config.initial_connection == 'full':
            for input_id in config.input_keys:
                for output_id in config.output_keys:
                    key = (input_id, output_id)
                    connection = config.connection_gene_type(key)
                    connection.init_attributes(config)
                    self.connections[key] = connection

    def create_node(self, config, node_id):
        node = config.node_gene_type(node_id)
        node.init_attributes(config)
        node.a = config.a_init_mean
        node.b = config.b_init_mean
        node.c = config.c_init_mean
        node.d = config.d_init_mean
        return node


class IZNeuron:
    """Sets up and simulates the iznn nodes (neurons)."""
    def __init__(self, bias, a, b, c, d, inputs):
        """
        a, b, c, d are the parameters of the Izhikevich model.

        :param float bias: The bias of the neuron.
        :param float a: The time-scale of the recovery variable.
        :param float b: The sensitivity of the recovery variable.
        :param float c: The after-spike reset value of the membrane potential.
        :param float d: The after-spike reset value of the recovery variable.
        :param inputs: A list of (input key, weight) pairs for incoming connections.
        :type inputs: list(tuple(int, float))
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.bias = bias
        self.inputs = inputs

        # Membrane potential (millivolts).
        self.v = self.c

        # Membrane recovery variable.
        self.u = self.b * self.v

        self.fired = 0.0
        self.current = self.bias
        self.last_spike = 0
        self.spike_train = []

    def advance(self, dt):
        """
        Advances simulation time by the given time step in milliseconds.

        v' = 0.04 * v^2 + 5v + 140 - u + I
        u' = a * (b * v - u)

        if v >= 30 then
            v <- c, u <- u + d
        """
        try:
            # Euler integration
            print(f"Current before advance: {self.v}")
            print(f"Current before: {self.current}")
            self.v += dt * (0.04 * self.v**2 + 5*self.v + 140 - self.u + self.current)
            print(f"Current after advance: {self.v}")
            self.u += dt * self.a * (self.b * self.v - self.u)
            
            self.fired = 0.0
            if self.v >= 30.0:
                self.fired = 1.0
                self.v = self.c
                self.u += self.d
                self.last_spike = 0
                self.spike_train.append(1)
            else:
                self.last_spike += 1
                self.spike_train.append(0)
                
            if len(self.spike_train) > 1000:
                self.spike_train.pop(0)
                
        except OverflowError:
            print("Overflow error in advance method")
            self.reset()

    def reset(self):
        """Resets all state variables."""
        self.v = self.c
        self.u = self.b * self.v
        self.fired = 0.0
        self.current = self.bias
        self.last_spike = 0
        self.spike_train = []


class IZNN:
    """Basic iznn network object."""
    def __init__(self, neurons, inputs, outputs):
        self.neurons = neurons
        self.inputs = inputs
        self.outputs = outputs
        self.input_values = {}

    def set_inputs(self, inputs):
        """Assign input voltages."""
        if len(inputs) != len(self.inputs):
            raise RuntimeError("Input size mismatch")
        for i, v in zip(self.inputs, inputs):
            self.input_values[i] = v

    def reset(self):
        """Reset all neurons to their default state."""
        for n in self.neurons.values():
            n.reset()

    def advance(self, dt):
        for n in self.neurons.values():
            n.current = n.bias
            for i, w in n.inputs:
                ineuron = self.neurons.get(i)
                if ineuron is not None:
                    ivalue = ineuron.fired
                else:
                    ivalue = self.input_values.get(i, 0)
                n.current += ivalue * w

        for n in self.neurons.values():
            n.advance(dt)

        return [self.neurons[i].fired for i in self.outputs]

    @staticmethod
    def create(genome, config):
        """ Receives a genome and returns its phenotype (a neural network). """
        genome_config = config.genome_config
        required = required_for_output(genome_config.input_keys, genome_config.output_keys, genome.connections)

        # Gather inputs and expressed connections.
        node_inputs = {}
        for cg in genome.connections.values():
            if not cg.enabled:
                continue

            i, o = cg.key
            if o not in required and i not in required:
                continue

            if o not in node_inputs:
                node_inputs[o] = [(i, cg.weight)]
            else:
                node_inputs[o].append((i, cg.weight))

        neurons = {}
        for node_key in required:
            ng = genome.nodes[node_key]
            inputs = node_inputs.get(node_key, [])
            neurons[node_key] = IZNeuron(ng.bias, ng.a, ng.b, ng.c, ng.d, inputs)

        return IZNN(neurons, genome_config.input_keys, genome_config.output_keys)