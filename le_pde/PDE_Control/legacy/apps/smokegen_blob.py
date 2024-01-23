from phi.fluidformat import *

# for scene in scenes("~/data/control/squares"):
#     scene.remove()

scenecount = 1000

for scene_index in range(scenecount):
    scene = new_scene("~/data/control/squares")
    start_x, start_y, end_x, end_y = np.random.randint(10, 110, 4)
    print(scene)
    scenelength = 32
    vx = (end_x-start_x) / float(scenelength)
    vy = (end_y-start_y) / float(scenelength)
    for frame in range(scenelength+1):
        time = frame / float(scenelength)
        array = np.zeros([128, 128, 1], np.float32)
        x = int(round(start_x * (1-time) + end_x * time))
        y = int(round(start_y * (1-time) + end_y * time))
        array[y:y+8, x:x+8, :] = 1
        velocity_array = np.empty([129, 129, 2], np.float32)
        velocity_array[...,0] = vx
        velocity_array[...,1] = vy
        write_sim_frame(scene.path, [array, velocity_array], ["Density", "Velocity"], frame)