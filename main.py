import numpy as np
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

if __name__ == '__main__':

    neuron_types = {
    'Regular Spiking': {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 8},
    'Fast Spiking': {'a': 0.1, 'b': 0.2, 'c': -65, 'd': 2},
    'Bursting': {'a': 0.02, 'b': 0.2, 'c': -50, 'd': 2}
}

time = 1000  
dt = 1.0  
current_values = np.arange(0, 201, 20) 

firing_rates = {key: [] for key in neuron_types.keys()}

for neuron_name, params in neuron_types.items():
    for I in current_values:
        neuron = IzhikevichNeuron(potencial=-65, recovery=params['b'] * -65, threshold=30, a=params['a'], b=params['b'], c=params['c'], d=params['d'])
        spikes = 0

        for t in range(int(time / dt)):
            neuron.step(I, dt)

            if neuron.v >= neuron.vt:
                spikes += 1

        firing_rate = spikes / (time / 1000) 
        firing_rates[neuron_name].append(firing_rate)

plt.figure(figsize=(8, 6))
for neuron_name, rates in firing_rates.items():
    plt.plot(current_values, rates, label=neuron_name, marker='o')

plt.xlabel('Injected Current (nA)')
plt.ylabel('Firing Rate (Hz)')
plt.title('Firing Rate vs. Injected Current for Different Neuron Types')
plt.legend()
plt.grid(True)
plt.show()
