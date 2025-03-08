import numpy as np
import neat
import math
import pygame
from iznn import *


gravity = 9.8
cart_force = 10.0
cart_mass = 1.0
pole_mass = 0.1
pole_length = 0.5 

max_time = 3600 * 30  # 30 min
time_step = 0.02  # 20 ms per iteration

position_limit = 2.4
angle_limit = 12 * (math.pi / 180)

# View
screen_width = 800
screen_height = 400
cart_width = 50
cart_height = 10
pole_pixel_length = 100

pygame.init()
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()

def simulate_cartpole(action, state):
    x, x_vel, theta, theta_vel = state # the 4 inputs
    force = cart_force if action == 1 else -cart_force

    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    temp = (force + pole_mass * pole_length * theta_vel**2 * sin_theta) / (cart_mass + pole_mass)
    theta_acc = (gravity * sin_theta - cos_theta * temp) / (pole_length * (4.0/3.0 - pole_mass * cos_theta**2 / (cart_mass + pole_mass)))
    x_acc = temp - pole_mass * pole_length * theta_acc * cos_theta / (cart_mass + pole_mass)

    x += x_vel * time_step
    x_vel += x_acc * time_step
    theta += theta_vel * time_step
    theta_vel += theta_acc * time_step

    return np.array([x, x_vel, theta, theta_vel])

def draw_cartpole(state, generation, avg_fitness, max_fitness, message=""):
    screen.fill((255, 255, 255))
    x, _, theta, _ = state
    x_pixels = int((x + position_limit) / (2 * position_limit) * screen_width)
    pygame.draw.rect(screen, (0, 0, 0), (x_pixels - cart_width // 2, screen_height // 2, cart_width, cart_height))

    pivot_x = x_pixels
    pivot_y = screen_height // 2
    end_x = pivot_x + int(pole_pixel_length * math.sin(theta))
    end_y = pivot_y - int(pole_pixel_length * math.cos(theta))
    pygame.draw.line(screen, (255, 0, 0), (pivot_x, pivot_y), (end_x, end_y), 5)

    font = pygame.font.SysFont('Arial', 20)
    text = font.render(f'Generation: {generation} | Avg Fitness: {avg_fitness:.2f} | Max Fitness: {max_fitness:.2f}', True, (0, 0, 0))
    screen.blit(text, (10, 10))

    if message:
        message_text = font.render(message, True, (255, 0, 0))
        screen.blit(message_text, (10, 40))

    pygame.display.flip()

generation_number = 0
def eval_genomes(genomes, config):
    global generation_number
    generation_number += 1
    total_fitness = 0
    max_fitness = 0
    for genome_id, genome in genomes:
        net = IZNN.create(genome, config)
        fitness = 0
        state = np.array([0, 0, 0.05, 0])
        for _ in range(int(max_time / time_step)):
            outputs = net.activate(state)  # Ativa a rede com o estado atual
            action = np.argmax(outputs)  # Obtém a ação com maior ativação
            state = simulate_cartpole(action, state)
            x, _, theta, _ = state
            if abs(x) > position_limit or abs(theta) > angle_limit:
                break
            fitness += (math.cos(theta) + 1)
        genome.fitness = fitness
        total_fitness += fitness
        max_fitness = max(max_fitness, fitness)
    avg_fitness = total_fitness / len(genomes)
    draw_cartpole(np.array([0, 0, 0.05, 0]), generation_number, avg_fitness, max_fitness)

def l2norm(x):
    return (sum(i**2 for i in x))**0.5

def run_neat(config_file):
    config = neat.Config(IZGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
    config.genome_config.aggregation_function_defs.add('my_l2norm_function', l2norm)


    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(eval_genomes, 300)
    print('\nBest genome:\n', winner)

    # Display the best genome
    state = np.array([0, 0, 0.05, 0])
    net = IZNN.create(winner, config)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = np.argmax(net.activate(state))
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
    config_path = 'config-feedforward.txt'
    run_neat(config_path)
