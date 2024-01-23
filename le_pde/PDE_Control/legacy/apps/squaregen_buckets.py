from phi.fluidformat import *

# for scene in scenes('~/data/control/rising-squares'):
#     scene.remove()

scenecount = 4000
scenelength = 16

for scene_index in range(scenecount):
    scene = new_scene('~/data/control/bounded-squares')
    start_x = np.random.randint(24, 128-24-11)
    start_y = np.random.randint(10, 102-11)
    end_x = np.random.randint(24, 128-24-11)
    end_y = np.random.randint(10, 102-11)
    print(scene)
    vx = (end_x-start_x) / float(scenelength)
    vy = (end_y-start_y) / float(scenelength)
    for frame in range(scenelength+1):
        time = frame / float(scenelength)
        array = np.zeros([128, 128, 1], np.float32)
        x = int(round(start_x * (1-time) + end_x * time))
        y = int(round(start_y * (1-time) + end_y * time))
        array[y:y+11, x:x+11, :] = 1
        velocity_array = np.empty([129, 129, 2], np.float32)
        velocity_array[...,0] = vx
        velocity_array[...,1] = vy
        write_sim_frame(scene.path, [array, velocity_array], ['Density', 'Velocity'], frame)



# for reference
def build_obstacles(sim):
    sim.set_obstacle((4, 20), (60, 0)) # Left --
    sim.set_obstacle((92, 4), (14, 128-22+2)) # Right |
    sim.set_obstacle((4, 128-60), (4, 30)) # Bottom ------
    sim.set_obstacle((38, 4), (14, 20-2)) # Left lower |
    sim.set_obstacle((34, 4), (72, 20-2)) # Left upper |
    # Buckets
    sim.set_obstacle((10, 2), (110-5, 20-1))
    sim.set_obstacle((10, 2), (110-5, 50-1))
    sim.set_obstacle((10, 2), (110-5, 80-1))
    sim.set_obstacle((10, 2), (110-5, 110-1))