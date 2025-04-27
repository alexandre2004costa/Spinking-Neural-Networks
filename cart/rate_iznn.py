import neat
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict

class SpikeMonitor:
    def __init__(self, network):
        """
        Inicializa um monitor de spikes para uma rede IZNN
        
        Args:
            network: A rede neural IZNN a ser monitorada
        """
        self.network = network
        self.spike_times = defaultdict(list)  # Armazena os tempos de spike para cada neurônio
        self.neuron_types = {}  # Armazena o tipo de cada neurônio (input, hidden, output)
        
        # Identifica os tipos de neurônios
        for neuron_id in network.neurons.keys():
            if neuron_id in network.inputs:
                self.neuron_types[neuron_id] = 'input'
            elif neuron_id in network.outputs:
                self.neuron_types[neuron_id] = 'output'
            else:
                self.neuron_types[neuron_id] = 'hidden'
    
    def reset(self):
        """Limpa todos os dados registrados"""
        self.spike_times.clear()
    
    def record(self, firing_neurons, time):
        """
        Registra quais neurônios dispararam em um determinado momento
        
        Args:
            firing_neurons: Conjunto de IDs dos neurônios que dispararam
            time: O tempo atual da simulação
        """
        for neuron_id in firing_neurons:
            self.spike_times[neuron_id].append(time)
    
    def plot_spikes(self, title="Spike Raster Plot", figsize=(12, 8), save_path=None):
        """
        Gera um gráfico de spikes ao longo do tempo
        
        Args:
            title: Título do gráfico
            figsize: Tamanho da figura (largura, altura)
            save_path: Caminho para salvar a figura (opcional)
        """
        # Organiza os neurônios por tipo (input, hidden, output)
        neurons_by_type = {
            'input': [],
            'hidden': [],
            'output': []
        }
        
        for neuron_id, neuron_type in self.neuron_types.items():
            if neuron_id in self.spike_times:  # Só inclui neurônios que dispararam
                neurons_by_type[neuron_type].append(neuron_id)
        
        # Calcula o número de neurônios que dispararam
        n_neurons = sum(len(neurons) for neurons in neurons_by_type.values())
        
        if n_neurons == 0:
            print("Nenhum spike detectado para visualizar.")
            return
        
        # Configuração da figura
        fig, ax = plt.subplots(figsize=figsize)
        
        # Cores para cada tipo de neurônio
        colors = {
            'input': 'blue',
            'hidden': 'green',
            'output': 'red'
        }
        
        # Mapeamento de IDs de neurônios para índices no eixo y
        neuron_indices = {}
        current_idx = 0
        
        # Plotagem dos spikes separados por tipo
        for neuron_type in ['input', 'hidden', 'output']:
            for neuron_id in sorted(neurons_by_type[neuron_type]):
                neuron_indices[neuron_id] = current_idx
                spike_times = self.spike_times[neuron_id]
                
                # Plotagem dos spikes como linhas verticais
                ax.vlines(spike_times, current_idx - 0.45, current_idx + 0.45, 
                          colors=colors[neuron_type], linewidth=1.5)
                
                current_idx += 1
            
            # Adiciona uma linha separadora entre tipos de neurônios
            if neurons_by_type[neuron_type] and neuron_type != 'output':
                ax.axhline(y=current_idx - 0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Configuração dos eixos
        ax.set_ylim(-0.5, n_neurons - 0.5)
        
        # Tempo máximo da simulação
        max_time = max(max(times) for times in self.spike_times.values()) if self.spike_times else 0
        ax.set_xlim(0, max_time)
        
        # Rótulos e layout
        ax.set_xlabel('Tempo (ms)')
        ax.set_ylabel('Neurônio')
        ax.set_title(title)
        
        # Criação das legendas para os tipos de neurônios
        legend_elements = [
            plt.Line2D([0], [0], color=color, lw=2, label=f'{neuron_type.capitalize()} Neurons')
            for neuron_type, color in colors.items()
            if neurons_by_type[neuron_type]  # Só inclui tipos que têm neurônios
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Adicionando rótulos no eixo y para identificar os neurônios
        yticks = []
        yticklabels = []
        
        for neuron_type in ['input', 'hidden', 'output']:
            if neurons_by_type[neuron_type]:
                # Calculando o índice médio para este tipo de neurônio
                indices = [neuron_indices[nid] for nid in neurons_by_type[neuron_type]]
                mid_idx = sum(indices) / len(indices)
                
                # Adicionando um rótulo para este grupo
                yticks.append(mid_idx)
                yticklabels.append(f"{neuron_type.capitalize()} ({len(indices)})")
        
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return fig, ax

    def plot_activity(self, window_size=10, title="Atividade Neural", figsize=(12, 6), save_path=None):
        """
        Gera um gráfico de atividade neural (contagem de spikes por janela de tempo)
        
        Args:
            window_size: Tamanho da janela temporal para contagem de spikes (ms)
            title: Título do gráfico
            figsize: Tamanho da figura (largura, altura)
            save_path: Caminho para salvar a figura (opcional)
        """
        if not self.spike_times:
            print("Nenhum spike detectado para visualizar.")
            return
        
        # Encontra o tempo máximo da simulação
        max_time = max(max(times) for times in self.spike_times.values())
        
        # Calcula o número de janelas
        n_windows = int(np.ceil(max_time / window_size))
        
        # Cria contadores para cada tipo de neurônio
        activity = {
            'input': np.zeros(n_windows),
            'hidden': np.zeros(n_windows),
            'output': np.zeros(n_windows)
        }
        
        # Conta spikes por janela para cada tipo de neurônio
        for neuron_id, times in self.spike_times.items():
            neuron_type = self.neuron_types[neuron_id]
            for t in times:
                window_idx = int(t / window_size)
                if window_idx < n_windows:
                    activity[neuron_type][window_idx] += 1
        
        # Normaliza pela quantidade de neurônios de cada tipo
        for neuron_type in ['input', 'hidden', 'output']:
            count = sum(1 for nid in self.neuron_types if self.neuron_types[nid] == neuron_type)
            if count > 0:
                activity[neuron_type] /= count
        
        # Configuração da figura
        fig, ax = plt.subplots(figsize=figsize)
        
        # Eixo x: centros das janelas temporais
        x = np.arange(n_windows) * window_size + window_size / 2
        
        # Plotagem para cada tipo de neurônio
        colors = {'input': 'blue', 'hidden': 'green', 'output': 'red'}
        for neuron_type in ['input', 'hidden', 'output']:
            ax.plot(x, activity[neuron_type], label=f'{neuron_type.capitalize()} Neurons', 
                     color=colors[neuron_type], linewidth=2)
        
        # Configuração dos eixos
        ax.set_xlabel('Tempo (ms)')
        ax.set_ylabel('Taxa de Disparo Média')
        ax.set_title(title)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return fig, ax
    
class RateIZNN(neat.iznn.IZNN):
    def __init__(self, neurons, inputs, outputs):
        super().__init__(neurons, inputs, outputs)
        self.simulation_steps = 100
        self.spike_trains = {i: [] for i in outputs}
        self.input_currents = {}  # Store converted input currents
        self.input_fired = {}  # Track input firing status
        self.monitor = SpikeMonitor(self)
        self.nowFiring = set()
        
    def set_inputs(self, inputs, I_min=0.0, I_max=10.0):
        """Store normalized inputs [0,1] for probability-based spike generation"""
        if len(inputs) != len(self.inputs):
            raise RuntimeError("Input size mismatch")
        for i, v in zip(self.inputs, inputs):
           self.input_values[i] = v
           self.input_currents[i] = I_min + v * I_max  # Scale input to current range
            
    def advance(self, dt):
        #self.monitor.reset()
        for n in self.neurons.values():
            n.spike_count = 0
        for o in self.outputs:
            self.spike_trains[o] = []

        input_firing_schedule = {}
        for i in self.inputs:
            num_fires = int(self.input_values[i] * self.simulation_steps)
            if num_fires > 0:
                firing_steps = set(np.linspace(0, self.simulation_steps-1, num_fires, dtype=int))
            else:
                firing_steps = set()
            input_firing_schedule[i] = firing_steps
        

        for t in range(self.simulation_steps):
            self.input_fired.clear()
            self.nowFiring.clear()
            for i in self.inputs: 
                #self.input_fired[i] = random.random() < self.input_values[i]
                self.input_fired[i] = t in input_firing_schedule[i]

            # --- Fase 1: Propagação dos inputs para os hidden ---
            for i, n in self.neurons.items():
                if i in self.outputs:
                    continue  # só tratamos hidden nesta fase

                n.current = n.bias + 0 # background

                for j, w in n.inputs:
                    if j in self.inputs:
                        if self.input_fired[j]:
                            n.current += w * self.input_currents[j]
                    else:
                        ineuron = self.neurons[j]
                        if ineuron is not None:
                            n.current += ineuron.fired * w * 10

            # Update hidden neurons
            for i, n in self.neurons.items():
                if i in self.outputs:
                    continue
                #print("Neuron", i)
                #print(n.current, n.v, n.bias)
                n.advance(dt)
                #print(n.v, n.u, n.fired, n.spike_count, n.current)
                if n.fired > 0:
                    n.spike_count += 1
                    self.nowFiring.add(i)

            # --- Fase 2: Propagação dos hidden para os output ---
            for i in self.outputs:
                n = self.neurons[i]
                n.current = n.bias + 0 # background
                #print(n.inputs)
                #print([n.fired for n in self.neurons.values()])
                for j, w in n.inputs:
                    if j in self.inputs:
                        if self.input_fired[j]:
                            n.current +=  w * self.input_currents[j]
                    else:
                        ineuron = self.neurons[j]
                        #print(ineuron)
                        if ineuron is not None:
                            n.current += ineuron.fired * w * 10
                            #print(ineuron.fired * w)
                            #print(n.current)

            # Update output neurons
            for i in self.outputs:
                n = self.neurons[i]
                #print(n.current)
                n.advance(dt)
                #print(n.v, n.u, n.fired, n.spike_count, n.current)
                if n.fired > 0:
                    #print("OUT SPIKED")
                    n.spike_count += 1
                    self.nowFiring.add(i)

           #print(self.nowFiring)
            #self.monitor.record(self.nowFiring, t)

        #self.monitor.plot_spikes(title="Atividade Neural - CartPole")
        #print(self.monitor.spike_times)
        #print("Spike counts:", [self.neurons[i].spike_count for i, j in self.neurons.items()])
        ##window_time = self.simulation_steps * self.dt
        return [self.neurons[i].spike_count for i in self.outputs]


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