from phi.fluidformat import *
from phi.flow import FluidSimulation

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

sim = FluidSimulation([128]*2)
build_obstacles(sim)


# for scene in scenes('~/data/control/rising-squares'):
#     scene.remove()

scenecount = 4000
scenelength = 16

for scene_index in range(scenecount):
    scene = new_scene('~/data/control/rising-squares')
    m = margin = 10
    start_x = np.random.randint(24+m, 128 - 24 - 11 - m)
    start_y = np.random.randint(8+m, 60)
    end_x = np.random.choice([35-5, 65 - 5, 99 - 5])
    end_y = 110-5
    print(scene)
    scene.write_sim_frame(sim._active_mask, "domain", 0)
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


