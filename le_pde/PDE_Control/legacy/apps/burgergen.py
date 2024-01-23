from phi.fluidformat import *
from phi.flow import advect, laplace
import pylab

# for scene in scenes("~/data/control/forced-burgerclash"):
#     scene.remove()

viscosity = 0.1
scenecount = 4000
steps = 32
size = 32
plot = False

print("Generating %d scenes with %d steps each. Viscosity=%f" % (scenecount, steps, viscosity))

for scene_index in range(scenecount):
    scene = new_scene("~/data/control/forced-burgerclash")
    print(scene)
    scene.copy_calling_script()

    u = u0 = np.zeros([1, size, 1], np.float32)
    idx = np.linspace(0, 1, size)
    leftloc = np.random.uniform(0.2, 0.4)
    leftamp = np.random.uniform(0, 3)
    leftsig = np.random.uniform(0.05, 0.15)
    rightloc = np.random.uniform(0.6, 0.8)
    rightamp = np.random.uniform(-3, 0)
    rightsig = np.random.uniform(0.05, 0.15)
    u[0,:,0] += leftamp * np.exp(-0.5 * (idx-leftloc)**2 / leftsig**2)
    u[0,:,0] += rightamp * np.exp(-0.5 * (idx-rightloc)**2 / rightsig**2)
    forceloc = np.random.uniform(0.4, 0.6)
    forceamp = np.random.uniform(-0.05, 0.05)
    forcesig = np.random.uniform(0.1, 0.4)
    force = np.zeros([1, size, 1], np.float32)
    force[0,:,0] += forceamp * np.exp(-0.5 * (idx-forceloc)**2 / forcesig**2)

    scene.write_sim_frame(u, "u", 0)
    scene.write_sim_frame(force, "force", 0)
    scene.properties = {"dimensions": [size], "viscosity": viscosity, "force": float(np.sum(force))}

    for step in range(1,steps+1):
        u += force
        u = advect(u, u)
        u += viscosity * laplace(u)
        scene.write_sim_frame(u, "u", step)

    if plot:
        pylab.plot(u0[0,:,0])
        pylab.plot(u[0,:,0])
        pylab.show()