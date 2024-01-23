# coding=utf-8
from __future__ import print_function
import sys
from phi.tf.model import *
from phi.control.control import *
from phi.control.iksm import *


def hierarchical_vec_pot(initial_value, levels=5):
    # return StaggeredGrid(tf.Variable(initial_value, dtype=tf.float32))

    if isinstance(levels, list):
        trainable = levels
        levels = len(levels)
    else:
        trainable = [True] * levels

    values = [initial_value]
    for i in range(levels-1):
        values.insert(0, downsample2x(values[0]))

    val = values[0]
    result = tf.Variable(values[0], dtype=tf.float32) * trainable[0]
    for i in range(1, levels):
        val = upsample2x(val)
        result = upsample2x(result)
        result = result + tf.Variable(values[i]-val, dtype=tf.float32) * trainable[i]
        val = values[i]
    return StaggeredGrid(result)


class SmokeSM(TFModel):
    def __init__(self):
        TFModel.__init__(self, "Overfit Smoke %d"%n, "Slow motion density generation CNN training",
                         learning_rate=1e-2, data_validation_fraction=0.1,
                         model_scope_name="vecpot")
        self.info("Loading sequence: %s" % init_path)
        self.info("Initialization: %s" % ("random" if random_velocities else "from sequence"))
        sim = self.sim = TFFluidSimulation([128] * 2, "open")
        vec_pots_init = np.load(os.path.join(init_path, "seq_vecpot.npy"))
        predicted_seq = np.load(os.path.join(init_path, "seq_pred.npy"))
        if load_real:
            initial_seq = np.load(os.path.join(init_path, "seq_real.npy"))
        else:
            initial_seq = predicted_seq
        target = self.target = predicted_seq[:,-1,...]
        if batch_size is not None:
            vec_pots_init = vec_pots_init[0:batch_size,...]
            initial_seq = initial_seq[0:batch_size,...]
        self.info("Sequence length is %d" % initial_seq.shape[1])
        self.info('Batch count: %d' % initial_seq.shape[0])
        self.info('Frames have shape: %s' % [initial_seq.shape[2:]])
        assert n == initial_seq.shape[1] - 1
        density = tf.constant(initial_seq[:,0,...], tf.float32)  # start density
        densities = self.densities = [ density ]
        velocities = self.velocities = []
        vec_pots = self.vec_pots = []
        trainable = self.trainable = [tf.placeholder(tf.float32, ()) for i in range(6)]
        # trainable = [self.editable_bool("Enable_Level_%d"%i, i == 0) for i in range(6)]

        force_losses = []
        forces = []

        for i in range(n):
            self.info("Building step %d" % i)
            with self.model_scope():
                if random_velocities:
                    vec_pot = hierarchical_vec_pot(np.random.randn(*vec_pots_init[:, i, ...].shape) * 0.01, trainable)
                else:
                    vec_pot = hierarchical_vec_pot(vec_pots_init[:, i, ...], trainable)
            vec_pots.append(vec_pot)
            velocity = vec_pot.curl().pad(0, 1, "symmetric")
            velocities.append(velocity)
            density = velocity.advect(density)
            densities.append(density)
            if i > 0:
                force = velocity - sim_velocity
                force_losses.append(l2_loss(force))
                forces.append(l1_loss(force, reduce_batches=False))
            sim_velocity = velocity.advect(velocity) + sim.buoyancy(density)

        # Density loss
        self.blurred_density_diff = normalize_to(density, target) - target
        self.blurred_density_diff_lowres = downsample2x(downsample2x(downsample2x(downsample2x(self.blurred_density_diff))))
        self.blurred_density_diff = blur(self.blurred_density_diff, 2.0, cutoff=6)
        self.blurred_density_diff_lowres = blur(self.blurred_density_diff_lowres, 2.0, cutoff=8)
        final_density_loss = l2_loss(self.blurred_density_diff) * self.editable_float("Scale_HighRes_Density", 1e3)
        final_density_loss_lowres = l2_loss(self.blurred_density_diff_lowres) * self.editable_float("Scale_LowRes_Density", 1e9)
        self.add_scalar("Density_Loss_HighRes", final_density_loss)
        self.add_scalar("Density_Loss_LowRes", final_density_loss_lowres)
        # Force loss
        force_loss = tf.add_n(force_losses) * self.editable_float("Force_Loss_Scale", 1e-2)
        self.add_scalar("Total_Force_Loss", force_loss)
        self.total_force = math.add(forces)
        self.minimizer("Unsupervised_Loss", force_loss + final_density_loss + final_density_loss_lowres)

        self.finalize_setup([])

        self.display_time = EditableInt("Frame Display", 0, (0, n))
        self.add_field("Density (Real)", lambda: self.view(densities[self.display_time]))
        self.add_field("Velocity", lambda: self.view(velocities[self.display_time]))
        self.add_field("Vector Potential", lambda: self.view(vec_pots[self.display_time].staggered))
        self.add_field("Target", lambda: target)

        self.time = start_step

        if "tensorboard" in sys.argv:
            from phi.tf.profiling import launch_tensorboard
            launch_tensorboard(self.scene.subpath("summary"), port=6005)

    def base_feed_dict(self):
        def is_trainable(level):
            if level == 0: return True
            if level == 1: return self.time >= target_iterations[0]
            if level == 2: return self.time >= target_iterations[1]
            if level == 3: return self.time >= target_iterations[2]
            if level == 4: return self.time >= target_iterations[3]
            if level == 5: return self.time >= target_iterations[4]
        return {self.trainable[i]: is_trainable(i) for i in range(len(self.trainable))}

    def step(self):
        if (random_velocities and self.time in target_iterations) or (not random_velocities and self.time % 1000 == 0):
            self.validate(create_checkpoint=True)
            self.action_plot_sequences()

        if auto_learning_rate:
            if random_velocities:
                self.float_learning_rate = 1e-1 if self.time < target_iterations[3] else 1e-2
            else:
                self.float_learning_rate = max(1e-1 * np.exp(-1e-2 * self.time), 1e-2)
        self.optimize(self.recent_optimizer_node)

    def action_plot_sequences(self):
        self.info("Computing frames...")
        densities = np.stack(self.view(self.densities, all_batches=True), 1)
        velocities = np.stack(self.view([v.staggered for v in self.velocities], all_batches=True), 1)
        vec_pots = np.stack(self.view([v.staggered for v in self.vec_pots], all_batches=True), 1)
        pred_densities = np.zeros_like(densities)
        pred_densities[:,-1,...] = self.target

        path = self.scene.subpath("seq_%d"%self.time, create=True)
        np.save(self.scene.subpath(join(path,"seq_pred")), pred_densities)
        np.save(self.scene.subpath(join(path,"seq_vecpot")), vec_pots)
        np.save(self.scene.subpath(join(path,"seq_real")), densities)
        np.save(self.scene.subpath(join(path,"seq_vel")), velocities)
        self.info("Saved sequence data to %s" % self.scene)
        return self


auto_learning_rate = True
start_step = 0
load_real = False

# Shapes
# init_path = "/home/holl/model/refine-smokesm-16/sim_000021" # Shapes staggered
# init_path = "/home/holl/model/refine-smokesm-16/sim_000022" # Shapes refined
init_path = '/home/holl/model/overfit-smoke-16/sim_000045/seq_26900'; load_real = True; start_step=26900; auto_learning_rate=False

n = 16
random_velocities = "initialize" not in sys.argv
batch_size = None

target_iterations = [0, 1500, 2500, 3000, 3500, 4000] if random_velocities else [0] * 10

# Natural Smoke
# init_path = "/home/holl/model/refine-smokesm-64/sim_000007" #"/home/holl/Downloads/Natural Smoke/Staggered"  # DiffPhys staggered
# init_path = "/home/holl/model/refine-smokesm-64/sim_000008"  # Supervised
# n = 64

app = SmokeSM()
app.action_plot_sequences()

if "autorun" in sys.argv: app.play()


app.show(display=("Density (Real)", "Target"), production=__name__ != "__main__")
