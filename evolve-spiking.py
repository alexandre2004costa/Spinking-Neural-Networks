import multiprocessing
import os

import neat
import visualize
import numpy as np
from cartPole import *  # Importa o teu CartPole SNN

# Parâmetros da simulação
time_limit = 500  # Número máximo de passos que o agente pode sobreviver
generation_number = 0

def simulate(genome, config):
    """Simula um episódio do CartPole usando um genoma."""
    net = neat.iznn.IZNN.create(genome, config)  # Cria a rede de Izhikevich
    state = np.array([0, 0, 0.05, 0])
    dt = net.get_time_step_msec()
    total_reward = 0
    
    for t in range(time_limit):
        net.set_inputs(state)  # Define os inputs da rede
        output = net.advance(dt)  # Avança a rede por um timestep
        action = 1 if output[0] > output[1] else 0  # Decide ação baseado nos outputs
        state = simulate_cartpole(action, state)
        x, _, theta, _ = state
        if abs(x) > position_limit or abs(theta) > angle_limit:
            break
        total_reward += (math.cos(theta) + 1)
    
    return total_reward

def eval_genome(genome, config):
    """Avalia um único genoma."""
    return simulate(genome, config)

def eval_genomes(genomes, config):
    """Avalia múltiplos genomas paralelamente."""
    global generation_number
    generation_number += 1
    for genome_id, genome in genomes:
        genome.fitness = simulate(genome, config)

def run(config_path):
    """Executa o NEAT para evoluir a rede SNN no CartPole."""
    config = neat.Config(neat.iznn.IZGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    config.output_nodes = 2  # Dois nós de saída para escolher entre esquerda e direita

    pop = neat.population.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    
    winner = pop.run(eval_genomes, 300)

    print('\nMelhor genoma:\n{!s}'.format(winner))
    state = np.array([0, 0, 0.05, 0])
    net = neat.iznn.IZNN.create(winner, config)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        net.set_inputs(state)  # Define os inputs da rede
        dt = net.get_time_step_msec()
        output = net.advance(dt)  # Avança a rede por um timestep
        action = 1 if output[0] > output[1] else 0  # Decide ação baseado nos outputs
        state = simulate_cartpole(action, state)
        x, _, theta, _ = state

        message = ""
        if abs(x) > position_limit or abs(theta) > angle_limit:
            message = "Failed! Restarting..."
            state = np.array([0, 0, 0.05, 0])

        draw_cartpole(state, generation_number, 0, 0, message)

        clock.tick(50)
    pygame.quit()
    
if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    run(os.path.join(local_dir, 'config-spiking.txt'))
