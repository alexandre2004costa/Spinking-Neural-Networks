import pygame
import math
import numpy as np

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

min_vals = np.array([-2.4, -12.0, -0.21, -12.0])
max_vals = np.array([2.4, 12.0, 0.21, 12.0])

def simulate_cartpole_cont(force, state):
    return simulate(force, state)

def simulate_cartpole(action, state):
    force = cart_force if action > 0 else -cart_force
    return simulate(force, state)

def simulate(force, state):
    x, x_vel, theta, theta_vel = state # the 4 inputs

    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    #print(force)
    temp = (force + pole_mass * pole_length * theta_vel**2 * sin_theta) / (cart_mass + pole_mass)
    theta_acc = (gravity * sin_theta - cos_theta * temp) / (pole_length * (4.0/3.0 - pole_mass * cos_theta**2 / (cart_mass + pole_mass)))
    x_acc = temp - pole_mass * pole_length * theta_acc * cos_theta / (cart_mass + pole_mass)

    x += x_vel * time_step
    x_vel += x_acc * time_step
    theta += theta_vel * time_step
    theta_vel += theta_acc * time_step

    return np.array([x, x_vel, theta, theta_vel])

def draw_cartpole(screen, state, generation, avg_fitness, max_fitness, message=""):

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
