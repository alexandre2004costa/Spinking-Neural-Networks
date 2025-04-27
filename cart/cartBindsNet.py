from bindsnet.network import Network
from bindsnet.network.nodes import Input, IzhikevichNodes
from bindsnet.network.topology import Connection
import torch

class CartpoleSNN(Network):
    def __init__(self, num_inputs=4, num_hidden=4, num_outputs=1):
        super().__init__()
        
        # Layers
        self.input_layer = Input(n=num_inputs)
        self.hidden_layer = IzhikevichNodes(n=num_hidden)
        self.output_layer = IzhikevichNodes(n=num_outputs)
        
        # Connections
        self.input_hidden = Connection(
            source=self.input_layer,
            target=self.hidden_layer,
            w=torch.randn(num_inputs, num_hidden)
        )
        self.hidden_output = Connection(
            source=self.hidden_layer,
            target=self.output_layer,
            w=torch.randn(num_hidden, num_outputs)
        )
        
        # Add to network
        self.add_layer(self.input_layer, name="Input")
        self.add_layer(self.hidden_layer, name="Hidden")
        self.add_layer(self.output_layer, name="Output")
        self.add_connection(self.input_hidden, source="Input", target="Hidden")
        self.add_connection(self.hidden_output, source="Hidden", target="Output")