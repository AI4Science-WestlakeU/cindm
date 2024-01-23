from phi.fluidformat import *
from phi.control.voxelutil import *
from phi.flow import FluidSimulation

for scene in scenes("~/data/control/specificshapes/i1"):
    scene.remove()

shapelib = scene_at("~/data/control/shapelib/sim_00000")
rectangle = shapelib.read_array("Shape", 7)
circle = shapelib.read_array("Shape", 1)
vbar = shapelib.read_array("Shape", 3)

sim = FluidSimulation([128]*2)

vbar_start = sim.zeros()
vbar_start[:, 50:50+32, 10:10+32, :] = rectangle
vbar_end = sim.zeros()
vbar_end[:, 36:36+32, 50:50+32, :] = vbar

circle_start = sim.zeros()
circle_start[:, 50:50+32, 90:90+32 :] = rectangle
circle_end = sim.zeros()
circle_end[:, 60:60+32, 50:50+32, :] = circle

circle_scene = new_scene("~/data/control/specificshapes/i1")
print(circle_scene)
circle_scene.write_sim_frame(circle_start, "Density", 0)
circle_scene.write_sim_frame(circle_end, "Density", 16)
circle_scene.write_sim_frame(circle_end, "Density", 8)

vbar_scene = new_scene("~/data/control/specificshapes/i1")
print(vbar_scene)
vbar_scene.write_sim_frame(vbar_start, "Density", 0)
vbar_scene.write_sim_frame(vbar_end, "Density", 16)
vbar_scene.write_sim_frame(vbar_end, "Density", 8)