import torch
import norse.torch as norse
from cartPole import *
import numpy as np

class CartPoleSNN(torch.nn.Module):
    def __init__(self, input_size=4, hidden_size=6, output_size=1):
        super().__init__()
        
        self.input_layer = norse.LIFParameters(
            v_th=torch.tensor(0.5),            # Lowered threshold
            v_leak=torch.tensor(0.0),          # Changed leak
            v_reset=torch.tensor(0.0),         # Changed reset
            method="super",
            alpha=torch.tensor(50.0),          # Increased alpha
        )
        
        # Create linear layers with bias
        self.linear1 = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.linear2 = torch.nn.Linear(hidden_size, output_size, bias=True)
        
        # Initialize weights with higher values
        torch.nn.init.uniform_(self.linear1.weight, a=0.0, b=1.0)
        torch.nn.init.uniform_(self.linear2.weight, a=0.0, b=1.0)
        
        # Create LIF cells with parameters
        self.lif1 = norse.LIFCell(p=self.input_layer)
        self.lif2 = norse.LIFCell(p=self.input_layer)
        
    def forward(self, x, state=None):
        if state is None:
            state = (None, None)
        
        # Apply weights and activation
        z = self.linear1(x)
        z = torch.relu(z)  # Add activation function
        z1, s1 = self.lif1(z, state[0])
        
        z = self.linear2(z1)
        z = torch.relu(z)  # Add activation function
        z2, s2 = self.lif2(z, state[1])
        
        return z2, (s1, s2)

def simulate(network, state, dt=0.02, sim_steps=50):  # Increased sim_steps
    with torch.no_grad():
        input_tensor = torch.FloatTensor(state).reshape(1, -1)
        # Scale inputs to larger range
        input_tensor = 2.0 * (input_tensor - input_tensor.min()) / (input_tensor.max() - input_tensor.min()) - 1.0
        
        spikes = 0
        hidden_state = None
        
        for _ in range(sim_steps):
            output, hidden_state = network(input_tensor, hidden_state)
            spikes += output.sum().item()
            
        return spikes / sim_steps

def main():
    snn = CartPoleSNN()
    state = np.array([0, 0, 0.05, 0])
    steps_balanced = 0
    
    while True:
        output = simulate(snn, state)
        print(output)
        action = 1 if output > 0.5 else 0
        state = simulate_cartpole(action, state)
        x, _, theta, _ = state
        
        if abs(x) > position_limit or abs(theta) > angle_limit:
            break
            
        steps_balanced += 1
        if steps_balanced >= 100000:
            break
    
    print(f"Steps balanced: {steps_balanced}")

if __name__ == "__main__":
    main()