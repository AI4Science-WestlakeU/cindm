from phi.fluidformat import *
from phi.control.voxelutil import *

shapelib = scene_at("~/data/control/shapelib/sim_000000")

scenecount = 1000

for scene_index in range(scenecount):
    scene = new_scene("~/data/control/simpleshapes")
    print(scene)
    start = single_shape([1,128,128,1], shapelib)
    end = single_shape([1,128,128,1], shapelib)
    write_sim_frame(scene.path, [start], ["Density"], 0)
    write_sim_frame(scene.path, [end], ["Density"], 8)
    write_sim_frame(scene.path, [end], ["Density"], 16)
    write_sim_frame(scene.path, [end], ["Density"], 32)
    write_sim_frame(scene.path, [end], ["Density"], 64)