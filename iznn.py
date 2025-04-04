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


    def __init__(self, key):
        super().__init__(key)
        #self.aggregation = "min"  

class IZGenome(DefaultGenome):
    def __init__(self, key):
        super().__init__(key)
        self.connections = {}
        self.nodes = {}

    @classmethod
    def parse_config(cls, param_dict):
        # These fields are required for NEAT-Python
        param_dict['initial_connection'] = 'partial_nodirect 0.5'
        # Using non-negative IDs: inputs 1-4, output 0, hidden nodes 5+
        param_dict['input_keys'] = [i+1 for i in range(4)]  # 1,2,3,4
        param_dict['output_keys'] = [0]  # Single output with ID 0
        param_dict['node_gene_type'] = IZNodeGene
        param_dict['connection_gene_type'] = DefaultConnectionGene
        if 'aggregation_function_defs' not in param_dict:
            param_dict['aggregation_function_defs'] = {
                'sum': sum,
                'max': max,
                'min': min
            }
        return DefaultGenomeConfig(param_dict)

    def configure_new(self, config):
        super().__init__(self.key)
        
        # Create input nodes (IDs 1-4)
        for i, k in enumerate(config.input_keys, 1):
            self.nodes[k] = IZNodeGene(k)
            self.nodes[k].bias = 0.0
            self.nodes[k].a = config.a_init_mean
            self.nodes[k].b = config.b_init_mean
            self.nodes[k].c = config.c_init_mean
            self.nodes[k].d = config.d_init_mean
        
        # Create output node (ID 0)
        k = config.output_keys[0]
        self.nodes[k] = IZNodeGene(k)
        self.nodes[k].bias = random.gauss(config.bias_init_mean, config.bias_init_stdev)
        self.nodes[k].a = config.a_init_mean
        self.nodes[k].b = config.b_init_mean
        self.nodes[k].c = config.c_init_mean
        self.nodes[k].d = config.d_init_mean

        # Hidden nodes will be added by mutation starting from ID 5
        # No need to create them here as they're handled by NEAT's mutation


class IZNeuron(object):
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

    def advance(self, dt_msec):
        """
        Advances simulation time by the given time step in milliseconds.

        v' = 0.04 * v^2 + 5v + 140 - u + I
        u' = a * (b * v - u)

        if v >= 30 then
            v <- c, u <- u + d
        """
        # TODO: Make the time step adjustable, and choose an appropriate
        # numerical integration method to maintain stability.
        # TODO: The need to catch overflows indicates that the current method is
        # not stable for all possible network configurations and states.
        try:
            self.v += 0.5 * dt_msec * (0.04 * self.v ** 2 + 5 * self.v + 140 - self.u + self.current)
            self.v += 0.5 * dt_msec * (0.04 * self.v ** 2 + 5 * self.v + 140 - self.u + self.current)
            self.u += dt_msec * self.a * (self.b * self.v - self.u)
        except OverflowError:
            print("OVERFLOW!!!!!!!!!!!!!!!!!!!!")
            # Reset without producing a spike.
            self.v = self.c
            self.u = self.b * self.v

        self.fired = 0.0
        if self.v > 30.0:
            # Output spike and reset.
            self.fired = 1.0
            self.v = self.c
            self.u += self.d


    def reset(self):
        """Resets all state variables."""
        self.v = self.c
        self.u = self.b * self.v
        self.fired = 0.0
        self.current = self.bias



class IZNN(object):
    """Basic iznn network object."""
    def __init__(self, neurons, inputs, outputs):
        self.neurons = neurons
        self.inputs = inputs
        self.outputs = outputs
        self.input_values = {}
        self.spike_count = 0  # Add spike counter

    def set_inputs(self, inputs):
        """Assign input voltages."""
        if len(inputs) != len(self.inputs):
            raise RuntimeError(
                "Number of inputs {0:d} does not match number of input nodes {1:d}".format(
                    len(inputs), len(self.inputs)))
        for i, v in zip(self.inputs, inputs):
            self.input_values[i] = v


    def reset(self):
        """Reset all neurons to their default state."""
        for n in self.neurons.values():
            n.reset()


    def get_time_step_msec(self):
        # pylint: disable=no-self-use
        # TODO: Investigate performance or numerical stability issues that may
        # result from using this hard-coded time step.
        return 0.1


    def advance(self, dt_msec):
        try:
            output_neuron = self.neurons[0]
            v_before = output_neuron.v
            
            # Calculate and log input currents
            total_current = 0
            for i, w in output_neuron.inputs:
                ineuron = self.neurons.get(i)
                if ineuron is not None:
                    ivalue = ineuron.fired
                else:
                    ivalue = self.input_values.get(i, 0)
                curr = ivalue * w
                total_current += curr
                if abs(curr) > 100:  # Log large currents
                    print(f"Large current from {i}: {curr:.2f}")
            
            # Original advance code
            for n in self.neurons.values():
                n.current = n.bias
                for i, w in n.inputs:
                    ineuron = self.neurons.get(i)
                    if ineuron is not None:
                        ivalue = ineuron.fired
                    else:
                        ivalue = self.input_values.get(i, 0)
                    n.current += ivalue * w

            # Debug voltage changes
            if abs(output_neuron.v - v_before) > 50:
                print(f"Large voltage change: {v_before:.2f} -> {output_neuron.v:.2f}")
                print(f"Current: {output_neuron.current:.2f}, Total input current: {total_current:.2f}")

            for n in self.neurons.values():
                n.advance(dt_msec)

            return [self.neurons[i].fired for i in self.outputs]
            
        except Exception as e:
            print(f"Error in IZNN advance: {str(e)}")
            raise

    def activate(self, inputs):
        self.set_inputs(inputs)
        dt = self.get_time_step_msec()
        return self.advance(dt)

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
                node_inputs[o].append((i, cg.weight))  # Fixed: removed extra list brackets

        neurons = {}
        for node_key in required:
            ng = genome.nodes[node_key]
            inputs = node_inputs.get(node_key, [])
            neurons[node_key] = IZNeuron(ng.bias, ng.a, ng.b, ng.c, ng.d, inputs)

        genome_config = config.genome_config
        return IZNN(neurons, genome_config.input_keys, genome_config.output_keys)