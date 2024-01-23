# coding=utf-8
from __future__ import print_function
import sys
from phi.tf.model import *
from phi.control.control import *
from phi.control.iksm import *


def ik(sim, initial_density, target_density, trainable=False):
    assert not trainable
    with tf.variable_scope("ik"):
        optimizable_velocity = ik_resnet(initial_density, target_density, trainable=trainable, training=False, reuse=tf.AUTO_REUSE)
    optimizable_velocity = optimizable_velocity.pad(0, 1, "symmetric").staggered
    zeros = math.zeros_like(optimizable_velocity)
    velocity = StaggeredGrid(tf.concat([optimizable_velocity[:, :, :30, :],
                                             zeros[:, :, 30:-30, :],
                                             optimizable_velocity[:, :, -30:, :]], axis=-2))
    velocity = sim.divergence_free(velocity, accuracy=1e-3)
    return velocity


def ik_resnet(initial_density, target_density, training=False, trainable=True, reuse=None):
    y = tf.concat([initial_density, target_density], axis=-1)
    y = tf.pad(y, [[0, 0], [0, 1 + 2 + 4 + 8 + 16 + 16], [0, 1 + 2 + 4 + 8 + 16 + 16], [0, 0]])
    resolutions = [y]
    for i in range(1, 6):  # 1/2, 1/4, 1/8, 1/16, 1/32
        y = tf.layers.conv2d(resolutions[0], 16, 2, strides=2, activation=tf.nn.relu, padding='valid',
                             name='downconv_%d' % i, trainable=trainable, reuse=reuse)
        for j, nb_channels in zip(range(3), [16, 16, 16]):
            y = residual_block(y, nb_channels, name='downrb_%d_%d' % (i, j), training=training, trainable=trainable,
                               reuse=reuse)
        resolutions.insert(0, y)

    y = tf.layers.conv2d(y, 16, 2, 1, activation=tf.nn.relu, padding='valid', name='centerconv_1', trainable=trainable,
                         reuse=reuse)

    fc_branch = tf.layers.conv2d(y, 4, 1, 1, activation=tf.nn.relu, padding='valid', name='fc_reduce', trainable=trainable,
                         reuse=reuse)
    fc_branch = tf.reshape(fc_branch, [-1, 64])
    fc_branch = tf.layers.dense(fc_branch, 64, activation=tf.nn.relu, name='fc_dense2', trainable=trainable, reuse=reuse)
    fc_branch = tf.reshape(fc_branch, [-1, 4, 4, 4])
    y = tf.concat([y[..., :-4], fc_branch], axis=-1)

    for j, nb_channels in zip(range(3), [16, 16, 16]):
        y = residual_block(y, nb_channels, name='centerrb_%d' % j, training=training, trainable=trainable, reuse=reuse)

    for i in range(1, len(resolutions)):
        y = upsample2x(y)
        res_in = resolutions[i][:, 0:y.shape[1], 0:y.shape[2], :]
        y = tf.concat([y, res_in], axis=-1)
        if i < len(resolutions) - 1:
            y = tf.pad(y, [[0, 0], [0, 1], [0, 1], [0, 0]], mode='SYMMETRIC')
            y = tf.layers.conv2d(y, 16, 2, 1, activation=tf.nn.relu, padding='valid', name='upconv_%d' % i,
                                 trainable=trainable, reuse=reuse)
            for j, nb_channels in zip(range(3), [16, 16, 16]):
                y = residual_block(y, nb_channels, 2, name='uprb_%d_%d' % (i, j), training=training,
                                   trainable=trainable, reuse=reuse)
        else:
            # Last iteration
            y = tf.pad(y, [[0, 0], [0, 1], [0, 1], [0, 0]], mode='SYMMETRIC')
            y = tf.layers.conv2d(y, 2, 2, 1, activation=None, padding='valid', name='upconv_%d' % i,
                                 trainable=trainable, reuse=reuse)

    return StaggeredGrid(y)  # This is the velocity




class GraphBuilder(PartitioningExecutor):

    def __init__(self, sim, true_densities, trainable_n, info, force_inference, ik_trainable=False):
        self.sim = sim
        self.true_densities = true_densities
        self.trainable_n = trainable_n
        self.info = info
        self.force_inference = force_inference
        self.ik_trainable = ik_trainable

    def create_frame(self, index, step_count):
        frame = PartitioningExecutor.create_frame(self, index, step_count)
        frame.true = self.true_densities[index]
        frame.pred = []
        frame.real = None
        frame.force = None
        frame.prev_force = None
        frame.jerk = None
        frame.density = None
        frame.velocity = None
        frame.prev_velocity = None
        if index == 0:
            frame.pred = [ frame.true ]
            frame.real = frame.true
            frame.density = frame.true
        elif index == step_count:
            frame.pred = [ frame.true ]
            frame.density = frame.true
        return frame

    def run_sm(self, n, initial_density, target_density):
        with tf.variable_scope("sm%d" % n):
            return sm_resnet(initial_density, target_density, trainable=n in self.trainable_n)

    def run_ik(self, initial_density, target_density):
        return ik(self.sim, initial_density, target_density, trainable=self.ik_trainable)

    def run_advect(self, velocity, density):
        return velocity.advect(density)

    def run_force(self, initial_velocity, target_velocity, initial_density, real_target_density):
        if self.force_inference == "forcenet":
            force, self.forcenet_path = forcenet(initial_density, initial_velocity, target_velocity)
        else:
            next_velocity = initial_velocity.advect(initial_velocity) + self.sim.buoyancy(real_target_density)
            if self.force_inference == "exact":
                next_velocity = self.sim.divergence_free(next_velocity)
            force = target_velocity - next_velocity
        return force

    def run_jerk(self, initial_velocity, initial_force, next_force):
        advected_force = initial_velocity.advect(initial_force)
        return next_force - advected_force

    def partition(self, n, initial_frame, target_frame, center_frame):
        PartitioningExecutor.partition(self, n, initial_frame, target_frame, center_frame)
        center_frame.density = self.run_sm(n, initial_frame.density, target_frame.density)
        center_frame.pred.append(center_frame.density)

    def execute_step(self, initial_frame, target_frame):
        PartitioningExecutor.execute_step(self, initial_frame, target_frame)
        initial_frame.velocity = target_frame.prev_velocity = self.run_ik(initial_frame.real, target_frame.pred[-1])
        target_frame.real = target_frame.density = self.run_advect(initial_frame.velocity, initial_frame.real)
        if initial_frame.prev_velocity is not None:
            initial_frame.force = self.run_force(initial_frame.prev_velocity, initial_frame.velocity, initial_frame.real, target_frame.real)
            target_frame.prev_force = initial_frame.force
        if initial_frame.prev_force is not None:
            initial_frame.jerk = self.run_jerk(initial_frame.prev_velocity, initial_frame.prev_force, initial_frame.force)

    def load_checkpoints(self, max_n, checkpoint_dict, preload_n):
        # Force
        if self.force_inference == "forcenet":
            self.info("Loading ForceNet checkpoint from %s..." % self.forcenet_path)
            self.sim.restore(self.forcenet_path, scope="ForceNet")
        # IK
        ik_checkpoint = os.path.expanduser(checkpoint_dict["IK"])
        self.info("Loading IK checkpoint from %s..." % ik_checkpoint)
        self.sim.restore(ik_checkpoint, scope="ik")
        # SM
        n = 2
        while n <= max_n:
            if n == max_n and not preload_n: return
            checkpoint_path = None
            i = n
            while not checkpoint_path:
                if "SM%d"%i in checkpoint_dict:
                    checkpoint_path = os.path.expanduser(checkpoint_dict["SM%d"%i])
                else:
                    i //= 2
            if i == n:
                self.info("Loading SM%d checkpoint from %s..." % (n, checkpoint_path))
                self.sim.restore(checkpoint_path, scope="sm%d" % n)
            else:
                self.info("Loading SM%d weights from SM%d checkpoint from %s..." % (n, i, checkpoint_path))
                self.sim.restore_new_scope(checkpoint_path, "sm%d" % i, "sm%d" % n)
            n *= 2


    def load_all_from(self, max_n, ik_checkpoint, sm_checkpoint, sm_n):
        # IK
        self.info("Loading IK checkpoint from %s..." % ik_checkpoint)
        self.sim.restore(ik_checkpoint, scope="ik")
        # SM
        n = 2
        while n <= max_n:
            source_n = sm_n(n) if callable(sm_n) else sm_n
            self.info("Loading SM%d weights from SM%d checkpoint from %s..." % (n, source_n, sm_checkpoint))
            self.sim.restore_new_scope(sm_checkpoint, "sm%d" % source_n, "sm%d" % n)
            n *= 2

    def lookup(self, array):
        return array



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
    pass


class SmokeSM(TFModel):
    def __init__(self):
        TFModel.__init__(self, "Refine Indirect SmokeSM %d"%n, "Slow motion density generation CNN training",
                         learning_rate=1e-4, data_validation_fraction=0.1,
                         training_batch_size=16, validation_batch_size=100,
                         model_scope_name="sm%d"%n)
        self.info("Mode: %s" % ("Graph" if use_graph else "Inference"))
        self.info("Preload: %s" % ("Yes" if preload else "No"))
        self.info("Divide Strategy: %s" % divide_strategy)
        self.info("Force: %s" % force_inference)
        self.info("Autorun: %s" % (("Supervised" if supervised else "Unsupervised") if autorun else "No"))
        self.setup()

        self.info("Setting up database...")
        fac = 2 if half_res else 1
        for i in range(n+1):
            field = BatchSelect(lambda len, i=i: range(fac*i, len-fac*n+fac*i), "Density")
            if half_res:
                field = transform.Downsample(field)
            self.database.add("density%d"%i, augment.AxisFlip(2, field))

        self.database.put_scenes(scenes('~/data/control/rising-squares'), per_scene_indices=range(n*fac+1), logf=self.info)

        self.finalize_setup([f.true for f in self.sequence if f.true is not None])

        build_obstacles(self.sim)

        # Load previously trained models
        ik_16_frame = "/home/holl/model/refine-smokeik-16/sim_000009/checkpoint_00043426"
        sm_supervised_squares = "/home/holl/model/supervised-squaresm/sim_000005/checkpoint_00003810"
        sm_diffphys = "/home/holl/model/refine-indirect-smokesm-16/sim_000001/checkpoint_00008806"  # build on ik_16, pretrained on sm_supervised_squares
        self.executor.load_all_from(n, ik_16_frame,
                                    sm_diffphys,
                                    lambda i: i)

        self.display_time = EditableInt("Frame Display", n//2, (0, n))

        self.add_field("Density (Ground Truth)", lambda: self.view_batch("density%d"%self.display_time))
        self.add_field("Density (Real)", lambda: self.view(self.sequence[self.display_time].real))
        self.add_field("Density (Predicted)", lambda: self.view(self.sequence[self.display_time].pred[0]))
        self.add_field("Velocity", lambda: self.view(self.sequence[self.display_time].velocity))
        self.add_field("Force", lambda: self.view(self.sequence[self.display_time].force))
        self.add_field("Jerk", lambda: self.view(self.sequence[self.display_time].jerk))
        self.add_field('Domain', self.sim.extended_fluid_mask)

        if not use_graph:
            self.step()

    def setup(self):
        # Simulation
        self.sim = TFFluidSimulation([128] * 2, DomainBoundary([(False, True), (False, False)]), force_use_masks=True)

        # Placeholders
        true_densities = [ None ] * (n+1)
        for i in [0, n//2, n]:
            true_densities[i] = self.sim.placeholder(name="density%d" % i)

        if use_graph:
            self.executor = GraphBuilder(self.sim, true_densities, trainable_n=range(n+2), ik_trainable=False, info=self.info, force_inference=force_inference)
        else:
            self.executor = EagerExecutor(self.sim, true_densities, self.info, force_inference)

        seq = self.sequence = get_divide_strategy(divide_strategy)(n, self.executor)

        if use_graph:
            self.sequence.execute()

            # Density loss
            self.blurred_density_diff = normalize_to(seq[-1].real, seq[-1].true) - seq[-1].true
            # for i in range(3):
            #     self.blurred_density_diff = downsample2x(self.blurred_density_diff)
            # self.blurred_density_diff = blur(self.blurred_density_diff, 4.0, cutoff=16)
            self.blurred_density_diff = blur(self.blurred_density_diff, 2.0, cutoff=16)
            final_density_loss = l2_loss(self.blurred_density_diff) * self.editable_float("Final_Density_Loss_Scale", 1e4) # 1e7 for 1/8 res 4px, 1e4 for 4px
            self.add_scalar("Final_Density_Loss", final_density_loss)
            # Force loss
            force_losses = []
            jerk_losses = []
            for frame in seq:
                if frame.force is not None:
                    force_losses.append(l2_loss(frame.force))
                    self.add_scalar("Force_%d"%frame.index, l1_loss(frame.force))
                if frame.jerk is not None:
                    jerk_losses.append(l2_loss(frame.jerk))
                    self.add_scalar("Jerk_%d"%frame.index, l1_loss(frame.jerk))
            force_loss = tf.add_n(force_losses) * self.editable_float("Force_Loss_Scale", 1e-2)
            self.add_scalar("Total_Force_Loss", force_loss)
            if jerk_losses:
                jerk_loss = tf.add_n(jerk_losses) * self.editable_float("Jerk_Loss_Scale", 1e-3)
                self.add_scalar("Total_Jerk_Loss", jerk_loss)
            else:
                jerk_loss = 0
            self.unsupervised_optim = self.minimizer("Unsupervised_Loss", force_loss + final_density_loss + jerk_loss)
            # Supervised loss
            supervised_loss = l2_loss((seq[n//2].pred[0] - seq[n//2].true) / (0.1+spatial_sum(seq[n//2].true))) * 1e6
            self.supervised_optim = self.minimizer("Supervised_Loss", supervised_loss)


    def step(self):
        if use_graph:
            self.tfstep(self.unsupervised_optim)
        else:
            self.executor.set_dict(self.feed_dict(self.val_iterator, False))
            self.sequence.execute()


    def action_plot_sequences(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        self.info("Computing frames...")
        real_densities = [self.executor.lookup(self.sequence[i].real) for i in range(len(self.sequence))]
        pred_densities = [self.executor.lookup(self.sequence[i].pred[-1]) for i in range(len(self.sequence))]
        real_velocities = [self.executor.lookup(self.sequence[i].velocity.staggered) for i in range(len(self.sequence) - 1)]
        if use_graph:
            data = self.view(real_densities + pred_densities + real_velocities, all_batches=True)
            real_densities = data[:len(real_densities)]
            pred_densities = data[len(real_densities):len(real_densities) + len(pred_densities)]
            real_velocities = data[len(real_densities) + len(pred_densities):]
        vmin = 0
        vmax = max(np.max(real_densities), np.max(pred_densities))

        np.save(self.scene.subpath("seq_pred"), np.stack(pred_densities, 1))
        np.save(self.scene.subpath("seq_real"), np.stack(real_densities, 1))
        np.save(self.scene.subpath("seq_vel"), np.stack(real_velocities, 1))

        for batch in range(real_densities[0].shape[0]):
            batchdir = os.path.join(self.get_image_dir(), "sequence_%d"%batch)
            self.info("Plotting batch batch %d to %s" % (batch, batchdir))
            os.mkdir(batchdir)
            for i in range(len(real_densities)):
                real_density = real_densities[i]
                pred_density = pred_densities[i]

                plt.figure(figsize=(20, 10))
                # Real
                plt.subplot2grid((1, 3), (0, 1))
                plt.title('Real')
                plt.imshow(real_density[batch, :, :, 0], interpolation="nearest", cmap="bwr", origin="lower", vmin=vmin, vmax=vmax)
                # Predicted
                plt.subplot2grid((1, 3), (0, 2))
                plt.title('Predicted')
                plt.imshow(pred_density[batch, :, :, 0], interpolation="nearest", cmap="bwr", origin="lower", vmin=vmin, vmax=vmax)
                # Save file
                plt.savefig(os.path.join(batchdir, "It_%d_Sequence_%d.png" % (self.time, i)))
                plt.close()

        self.info("Saved all sequences to %s" % self.get_image_dir())
        return self

if "help" in sys.argv or "-help" in sys.argv or "--help" in sys.argv:
    print("First argument: n (integer)")
    print("Keywords: inference, preload, all_frames, adaptive, autorun, unsupervised, exact_force, forcenet")
    exit(0)
if len(sys.argv) >= 2:
    n = int(sys.argv[1])
else:
    n = 2
use_graph = "inference" not in sys.argv
preload = True

divide_strategy = "adaptive" if "adaptive" in sys.argv else "binary"
autorun = "autorun" in sys.argv
supervised = "supervised" in sys.argv
force_inference = "exact" if "exact_force" in sys.argv else "advection"
half_res = "half_res" in sys.argv
if "forcenet" in sys.argv: force_inference = "forcenet"

app = SmokeSM()
if autorun: app.play()

app.prepare()
app.action_plot_sequences()

# app.show(display=("Density (Predicted)", "Density (Ground Truth)"), depth=31, production=__name__ != "__main__")