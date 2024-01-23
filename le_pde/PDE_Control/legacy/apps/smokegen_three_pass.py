from __future__ import print_function
from phi.tf.flow import *
from phi.model import *


class SmokeGen3P(FieldSequenceModel):

    def __init__(self, size, steps=70):
        FieldSequenceModel.__init__(self, "SmokeGen3P",
                                    "Random smoke with backwards advection",
                                    record_data=True,
                                    base_dir="SmokeIK",
                                    recorded_fields=("Density", "Velocity"),
                                    summary="random3p_"+"x".join([str(i) for i in size]))
        self.add_field("Density", lambda: self.density_data)
        self.add_field("Initial Density", lambda: self.initial_density)
        self.add_field("Velocity", lambda: self.velocity_data)

        self.margin = 10

        self.sim = TFFluidSimulation(size, "open", 1)

        self.velocity_in = self.sim.placeholder("staggered", "velocity")
        self.density_in = self.sim.placeholder(1, "density")
        self.sim_step(self.velocity_in, self.density_in, self.sim)

        self.v_scale = EditableFloat("Velocity Base Scale", 0.5, (0, 4), log_scale=False)
        self.v_scale_rnd = EditableFloat("Velocity Randomization", 0.5, (0, 1), log_scale=False)
        self.v_falloff = EditableFloat("Velocity Power Spectrum Falloff", 0.9, (0, 1), log_scale=False)
        self.steps_per_scene = steps
        self.prepare()
        self.sim.initialize_variables()
        self.action_reset()

    def sim_step(self, velocity, density, sim):
        self.next_density = velocity.advect(density)
        self.next_velocity = sim.divergence_free(velocity.advect(velocity) + sim.buoyancy(density))

    def step(self):
        if self.simpass == 0:
            # Forward propagation of full density field
            self.velocity_data, self.density_data = self.sim.run([self.next_velocity, self.next_density], feed_dict=self.feed())
            self.velocity_cache.append(self.velocity_data)
            if self.time == self.steps_per_scene:
                self.info("Finished pass 1 (forward simulation) for scene %s"%self.scene)
                self.simpass = 1
                self.recorded_fields = ()
                density_data = np.zeros_like(self.density_data)
                valid_region = [slice(None)] + [slice(self.margin,-self.margin)] * self.sim.rank + [slice(None)]
                density_data[valid_region] = self.density_data[valid_region]
                self.density_data = density_data
        elif self.simpass == 1:
            # Backpropagate density as mask
            self.time -= 2
            self.density_data = self.velocity_cache[self.time].advect(self.density_data, dt=-1)
            if self.time == 0:
                self.info("Finished pass 2 (backward advection) for scene %s"%self.scene)
                self.recorded_fields = ("Density",)
                self.density_data = self.initial_density * (self.density_data >= 0.04)
                self.simpass = 2
        elif self.simpass == 2:
            # Forward propagate masked density
            self.density_data = self.velocity_cache[self.time].advect(self.density_data)
            if self.time == self.steps_per_scene:
                self.info("Finished pass 3 (forward advection) for scene %s"%self.scene)
                self.action_reset()
                self.new_scene()


    def feed(self):
        return {self.velocity_in: self.velocity_data, self.density_in: self.density_data}

    def action_reset(self):
        v_scale = self.v_scale * (1 + (np.random.rand()-0.5) * 2)
        self.info("Creating scene %s with v_scale=%f, v_falloff=%f"%(self.scene, v_scale, self.v_falloff))
        self.add_custom_properties({"velocity_scale": v_scale, "velocity_falloff": self.v_falloff})

        # Velocity
        size = [1 for dim in self.sim.dimensions]
        rand = np.zeros([1]*(len(size)+1)+[len(size)])
        i = 0
        while size[0] < self.sim.dimensions[0]:
            rand = upsample2x(rand)
            size = [s * 2 for s in size]
            rand += np.random.randn(*([1]+size+[len(size)])) * v_scale * self.v_falloff**i
            i += 1
        rand = math.pad(rand, [[0,0]]+ [[0,1]]*self.sim.rank + [[0,0]], "symmetric")
        self.velocity_data = StaggeredGrid(rand)

        # Density
        density_data = upsample2x(upsample2x(np.random.rand(*self.sim.shape(scale=1.0 / 4)))) \
                       + upsample2x(upsample2x(upsample2x(np.random.rand(*self.sim.shape(scale=1.0 / 8))))) \
                       + upsample2x(upsample2x(upsample2x(upsample2x(np.random.rand(*self.sim.shape(scale=1.0 / 16))))))
        density_data = np.minimum(np.maximum(0, density_data * 0.66 - 1), 1)
        self.density_data = self.sim.zeros()
        valid_density_range = [slice(None)] + [slice(self.margin, -self.margin)] * self.sim.rank + [slice(None)]
        self.density_data[valid_density_range] = density_data[valid_density_range]
        self.initial_density = self.density_data

        self.velocity_cache = [ self.velocity_data ]
        self.simpass = 0
        self.time = 0
        self.recorded_fields = ("Velocity",)


app = SmokeGen3P([128] * 2).play().show(display=("Density", "Initial Density"), depth=31, production=__name__ != "__main__")
