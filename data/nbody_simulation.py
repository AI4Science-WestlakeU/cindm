#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import numpy as np
import pprint as pp
import pymunk
import pygame
import random
from tqdm.auto import tqdm
from utils import make_dir
import imageio
import pdb

# In[ ]:


# Configuration
parser = argparse.ArgumentParser(description='Train EBM model')
parser.add_argument('--n_bodies', default=2, type=int,
                    help='Number of bodies')
parser.add_argument('--n_simulations', default=2, type=int,
                    help='Number of simulations')
parser.add_argument('--vx', default=100, type=int,
                    help='max speed of balls in x-axis')
parser.add_argument('--vy', default=100, type=int,
                    help='max speed of balls in y-axis')
# try:
#     get_ipython().run_line_magic('matplotlib', 'inline')
#     %load_ext autoreload
#     %autoreload 2
#     is_jupyter = True
#     FLAGS = parser.parse_args([])
#     FLAGS.n_bodies = 2
# except:
#     FLAGS = parser.parse_args()
FLAGS = parser.parse_args()
pp.pprint(FLAGS.__dict__)

width, height = 200, 200
n_bodies = FLAGS.n_bodies
radius = 20
mass = 1
n_simulations = FLAGS.n_simulations
n_steps = 1000
is_visualize = True
filename = f'dataset/nbody_dataset/nbody-{n_bodies}/speed-{FLAGS.vx}/trajectory_balls_{n_bodies}_simu_{n_simulations}_steps_{n_steps}.npy'
make_dir(filename)
print(f"Save file at {filename}.")

def add_body(space):
    inertia = pymunk.moment_for_circle(mass, 0, radius)
    body = pymunk.Body(mass, inertia)
    x = random.randint(radius, width - radius)
    y = random.randint(radius, height - radius)
    body.position = x, y

    # Assign a random initial velocity
    vx = random.uniform(-100, 100)
    vy = random.uniform(-100, 100)
    body.velocity = vx, vy

    shape = pymunk.Circle(body, radius)
    shape.elasticity = 1.0
    shape.friction = 0.0
    space.add(body, shape)
    return body

def add_walls(space):
    walls = [
        pymunk.Segment(space.static_body, (0, 0), (0, height), 1),
        pymunk.Segment(space.static_body, (0, height), (width, height), 1),
        pymunk.Segment(space.static_body, (width, height), (width, 0), 1),
        pymunk.Segment(space.static_body, (width, 0), (0, 0), 1)
    ]
    for wall in walls:
        wall.elasticity = 1.0
        wall.friction = 0.0
        space.add(wall)

def draw_bodies(screen, bodies):
    for body in bodies:
        x, y = body.position
        color = pygame.Color(body.color)
        pygame.draw.circle(screen, color, (int(x), int(y)), radius)

def run_simulation(space, bodies, n_steps, clock,n_simulation ,screen=None):
    videofile=f'dataset/nbody_dataset/nbody-{n_bodies}/speed-{FLAGS.vx}/trajectory_balls_{n_bodies}_simu_{n_simulation}_steps_{n_steps}.gif'
    position_data = np.zeros((n_steps, len(bodies), 4))
    frames=[]
    for i in range(n_steps):
        if screen is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None

        for j, body in enumerate(bodies):
            position_data[i, j, :2] = body.position
            position_data[i, j, 2:] = body.velocity

        if screen is not None:
            screen.fill((255, 255, 255))
            draw_bodies(screen, bodies)  # Draw the bodies with colors

        space.step(1/60.0)
        if screen is not None:
            pygame.display.flip()
            screen_data = pygame.surfarray.array3d(screen)
            frames.append(screen_data)
        clock.tick(60)
    if screen is not None:
        imageio.mimsave(videofile, frames)
    return position_data

def main():
    pygame.init()
    if is_visualize:
        screen = pygame.display.set_mode((width, height))
    else:
        screen = None
    frames=[]
    clock = pygame.time.Clock()

    all_position_data = []

    for sim in tqdm(range(n_simulations)):
        print(f"simu: {sim}")
        space = pymunk.Space()
        space.gravity = (0, 0)

        add_walls(space)  # Add the walls

        bodies = [add_body(space) for _ in range(n_bodies)]

        # Assign a random color to each body
        for body in bodies:
            body.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        position_data = run_simulation(space, bodies, n_steps, clock,sim,screen=screen)
        if position_data is not None:
            all_position_data.append(position_data)
        else:
            break
        # Save the position data to an .npy file
        all_position_data_save = np.array(all_position_data)
        whether_saved=np.save(filename, all_position_data_save) ##[n_simulations, n_steps, n_bodies, 4 ] 4 inculde (x,y,vx,vy)：
        if whether_saved:    
            print("Successfully saved data!")
        else:
            print("Failed to save data")


    pygame.quit()

if __name__ == "__main__":
    main()
    #The following is to process data with adding more features for the dataset,just for two bodies
    # import torch
    # import math
    # import numpy as np
    # width=200
    # height=200
    # filename2 = f'dataset/nbody_dataset/nbody-{n_bodies}/speed-{FLAGS.vx}/trajectory_balls_{n_bodies}_simu_{2}_steps_{1000}_processed.npy'
    # dataset_init=np.load(filename)
    # dataset_init_tensor=torch.tensor(dataset_init)
    # dataset_final=torch.ones((dataset_init_tensor.shape[0],dataset_init_tensor.shape[1],dataset_init_tensor.shape[2],11),dtype=float)
    # for i in range(dataset_final.shape[0]):
    #     for j in range(dataset_final.shape[1]):
    #         for k in range(dataset_final.shape[2]):
    #             dataset_final[i][j][k][0]=dataset_init_tensor[i][j][k][0]
    #             dataset_final[i][j][k][1]=dataset_init_tensor[i][j][k][1]
    #             dataset_final[i][j][k][2]=dataset_init_tensor[i][j][k][2]
    #             dataset_final[i][j][k][3]=dataset_init_tensor[i][j][k][3]
    #             dataset_final[i][j][k][4]=dataset_init_tensor[i][j][k][0]-dataset_init_tensor[i][j][1-k][0]
    #             dataset_final[i][j][k][5]=dataset_init_tensor[i][j][k][1]-dataset_init_tensor[i][j][1-k][1]
    #             #distance between body1 and body2
    #             dataset_final[i][j][k][6]=torch.sqrt((dataset_init_tensor[i][j][k][0]-dataset_init_tensor[i][j][1-k][0])*(dataset_init_tensor[i][j][k][0]-dataset_init_tensor[i][j][1-k][0])+(dataset_init_tensor[i][j][k][1]-dataset_init_tensor[i][j][1-k][1])*(dataset_init_tensor[i][j][k][1]-dataset_init_tensor[i][j][1-k][1]))
    #             # distance between body and left wall
    #             dataset_final[i][j][k][7]=torch.abs(dataset_init_tensor[i][j][k][0])
    #             # distance between body and up wall
    #             dataset_final[i][j][k][8]=torch.abs(dataset_init_tensor[i][j][k][1])
    #             # distance between body and right wall
    #             dataset_final[i][j][k][9]=torch.abs(width-dataset_init_tensor[i][j][k][0])
    #             # distance between body and down wall
    #             dataset_final[i][j][k][10]=torch.abs(height-dataset_init_tensor[i][j][k][1])
    # dataset_final_np=np.array(dataset_final)
    # np.save(filename2,dataset_final_np)

