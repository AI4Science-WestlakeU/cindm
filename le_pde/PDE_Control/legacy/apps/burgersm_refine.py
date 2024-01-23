# coding=utf-8
from __future__ import print_function
import sys
from phi.tf.model import *
from phi.control.control import *
from phi.control.iksm import *


def sm_resnet(initial, target, training=False, trainable=True, reuse=tf.AUTO_REUSE):
    y = tf.concat([initial, target], axis=-1)
    downres_padding = sum([2**i for i in range(5)])
    y = tf.pad(y, [[0,0], [0,downres_padding], [0,0]])
    resolutions = [ y ]
    for i, filters in enumerate([4, 8, 16, 16, 16]):
        y = tf.layers.conv1d(resolutions[0], filters, 2, strides=2, activation=tf.nn.relu, padding="valid", name="downconv_%d"%i, trainable=trainable, reuse=reuse)
        for j in range(2):
            y = residual_block_1d(y, filters, name="downrb_%d_%d" % (i,j), training=training, trainable=trainable, reuse=reuse)
        resolutions.insert(0, y)

    for j, nb_channels in enumerate([16, 16, 16]):
        y = residual_block_1d(y, nb_channels, name="centerrb_%d" % j, training=training, trainable=trainable, reuse=reuse)

    for i, resolution_data in enumerate(resolutions[1:]):
        y = upsample2x(y)
        res_in = resolution_data[:, 0:y.shape[1], :]
        y = tf.concat([y, res_in], axis=-1)
        if i < len(resolutions)-2:
            y = tf.pad(y, [[0, 0], [0, 1], [0, 0]], mode="SYMMETRIC")
            y = tf.layers.conv1d(y, 16, 2, 1, activation=tf.nn.relu, padding="valid", name="upconv_%d" % i, trainable=trainable, reuse=reuse)
            for j, nb_channels in enumerate([16, 16]):
                y = residual_block_1d(y, nb_channels, 2, name="uprb_%d_%d" % (i, j), training=training, trainable=trainable, reuse=reuse)
        else:
            # Last iteration
            y = tf.pad(y, [[0,0], [0,1], [0,0]], mode="SYMMETRIC")
            y = tf.layers.conv1d(y, 1, 2, 1, activation=None, padding="valid", name="upconv_%d"%i, trainable=trainable, reuse=reuse)

    return y


class GraphBuilder(PartitioningExecutor):

    def __init__(self, sim, true_densities, trainable_n, info, force_inference):
        self.sim = sim
        self.true_densities = true_densities
        self.trainable_n = trainable_n
        self.info = info
        self.force_inference = force_inference

    def create_frame(self, index, step_count):
        frame = PartitioningExecutor.create_frame(self, index, step_count)
        frame.true = self.true_densities[index]
        frame.u = []
        frame.force = None
        frame.prev_force = None
        frame.jerk = None
        if index == 0 or index == step_count:
            frame.u = [ frame.true ]
        return frame

    def run_sm(self, n, initial_density, target_density):
        with tf.variable_scope("sm%d" % n):
            return sm_resnet(initial_density, target_density, trainable=n in self.trainable_n)

    def run_advect(self, u):
        u = advect(u, u)
        u += viscosity * laplace(u)
        return u

    def run_jerk(self, u, initial_force, next_force):
        advected_force = advect(initial_force, u)
        return next_force - advected_force

    def partition(self, n, initial_frame, target_frame, center_frame):
        PartitioningExecutor.partition(self, n, initial_frame, target_frame, center_frame)
        center_frame.u.append(self.run_sm(n, initial_frame.u[-1], target_frame.u[-1]))

    def execute_step(self, initial_frame, target_frame):
        PartitioningExecutor.execute_step(self, initial_frame, target_frame)
        target_frame.prev_force = initial_frame.force = target_frame.u[-1] - self.run_advect(initial_frame.u[-1])
        if initial_frame.prev_force is not None:
            initial_frame.jerk = self.run_jerk(initial_frame.u[-1], initial_frame.prev_force, initial_frame.force)


    def load_checkpoints(self, max_n, checkpoint_dict, preload_n):
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


    def load_all_from(self, max_n, sm_checkpoint, sm_n):
        n = 2
        while n <= max_n:
            source_n = sm_n(n) if callable(sm_n) else sm_n
            self.info("Loading SM%d weights from SM%d checkpoint from %s..." % (n, source_n, sm_checkpoint))
            self.sim.restore_new_scope(sm_checkpoint, "sm%d" % source_n, "sm%d" % n)
            n *= 2

    def lookup(self, array):
        return array



class BurgerSM(TFModel):
    def __init__(self):
        TFModel.__init__(self, "Refine-BurgerSM %d"%n, "Slow motion density generation CNN training",
                         learning_rate=1e-4, data_validation_fraction=0.1,
                         model_scope_name="sm%d"%n)
        self.info("Mode: %s" % ("Graph" if use_graph else "Inference"))
        self.info("Preload: %s" % ("Yes" if preload else "No"))
        self.info("Load frames: %s" % ("All" if load_all_frames else "Start/End/Center"))
        self.info("Divide Strategy: %s" % divide_strategy)
        self.info("Force: %s" % force_inference)
        self.info("Autorun: %s" % (("Supervised" if supervised else "Unsupervised") if autorun else "No"))
        self.setup()

        self.info("Setting up database...")
        for i in range(n+1):
            field = BatchSelect(lambda len, i=i: range(i, len-n+i), "u")
            self.database.add("u%d"%i, augment.AxisFlip(2, field))
        zeros = [0]*33*400
        self.database.add("force", BatchSelect(zeros, "force"))
        self.database.put_scenes(scenes("~/data/control/forced-burgerclash"), per_scene_indices=range(33), logf=self.info)
        self.finalize_setup([f.true for f in self.sequence if f.true is not None] + [self.force_placeholder])

        # Load checkpoints
        # unsupervised_models = {
        #     "SM2": "/home/holl/model/smokesm-2/sim_000232/checkpoint_00062648",
        #     "SM4": "/home/holl/model/smokesm-4/sim_000023/checkpoint_00016073",
        #     "SM8": "/home/holl/model/smokesm-8/sim_000011/checkpoint_00012925",
        #     "SM16": "/home/holl/model/smokesm-16/sim_000016/checkpoint_00002833",
        #     "SM32": "/home/holl/model/burgersm-32/sim_000000/checkpoint_00002250"
        # }
        # supervised_models = {
        #     "SM2": "/home/holl/model/burgersm-2/sim_000000/checkpoint_00006886",
        #     "SM4": "/home/holl/model/burgersm-4/sim_000000/checkpoint_00010593",
        #     "SM8": "/home/holl/model/burgersm-8/sim_000000/checkpoint_00009691",
        #     "SM16": "/home/holl/model/burgersm-16/sim_000000/checkpoint_00004999",
        #     "SM32": "/home/holl/model/burgersm-32/sim_000001/checkpoint_00001002",
        # }

        supervised_models = {  # with driving force
            "SM2": "/home/holl/model/burgersm-2/sim_000002/checkpoint_00017804",
            "SM4": "/home/holl/model/burgersm-4/sim_000001/checkpoint_00046434",
            "SM8": "/home/holl/model/burgersm-8/sim_000001/checkpoint_00015593",
            "SM16": "/home/holl/model/burgersm-16/sim_000001/checkpoint_00078263",
            "SM32": "/home/holl/model/burgersm-32/sim_000002/checkpoint_00054755",
        }
        # self.executor.load_checkpoints(n, supervised_models, preload_n=preload)

        self.executor.load_all_from(n, "/home/holl/model/refine-burgersm-32/sim_000008/checkpoint_00587946", lambda i: i)  # binary tree joint

        self.executor.load_all_from(n, "/home/holl/model/refine-burgersm-32/sim_000012/checkpoint_00235044", lambda i: i)  # unsupervised interleaved training


        self.value_supervised = supervised

        self.display_time = EditableInt("Frame Display", n//2, (0, n))

        self.add_field("u (Ground Truth)", lambda: self.view_batch("u%d"%self.display_time))
        self.add_field("u (Predicted)", lambda: self.view(self.sequence[self.display_time].u[0]))
        self.add_field("Velocity", lambda: self.view(self.sequence[self.display_time].velocity))
        self.add_field("Force", lambda: self.view(self.sequence[self.display_time].force))
        self.add_field("Jerk", lambda: self.view(self.sequence[self.display_time].jerk))

        if not use_graph:
            self.step()

    def setup(self):
        # Factors
        self.is_supervised = tf.placeholder(tf.float32, (), "is_supervised")
        self.add_scalar("Is_Supervised", self.is_supervised)
        # Simulation
        self.sim = TFFluidSimulation([32], "open", 100)

        # Placeholders
        if load_all_frames:
            true_densities = [self.sim.placeholder(name="u%d" % i) for i in range(n + 1)]
        else:
            true_densities = [ None ] * (n+1)
            for i in [0, n//2, n]:
                true_densities[i] = self.sim.placeholder(name="u%d" % i)

        self.force_placeholder = self.sim.placeholder(name="force")
        self.gt_force = l1_loss(self.force_placeholder, reduce_batches=False) * n

        self.executor = GraphBuilder(self.sim, true_densities, trainable_n=range(n+1), info=self.info, force_inference=force_inference)

        seq = self.sequence = get_divide_strategy(divide_strategy)(n, self.executor)

        if use_graph:
            self.sequence.execute()

            # Force loss
            force_losses = []
            forces = []
            jerk_losses = []
            for frame in seq:
                if frame.force is not None:
                    force_losses.append(l2_loss(frame.force))
                    forces.append(l1_loss(frame.force, reduce_batches=False))
                    self.add_scalar("Force_%d"%frame.index, l1_loss(frame.force))
                if frame.jerk is not None:
                    jerk_losses.append(l2_loss(frame.jerk))
                    self.add_scalar("Jerk_%d"%frame.index, l1_loss(frame.jerk))
            force_loss = tf.add_n(force_losses) * self.editable_float("Force_Loss_Scale", 1e-1)
            self.add_scalar("Total_Force_Loss", force_loss)
            self.total_force = math.add(forces)
            self.baseline_force = l1_loss(seq[0].true - seq[-1].true, reduce_batches=False)
            if jerk_losses:
                jerk_loss = tf.add_n(jerk_losses) * self.editable_float("Jerk_Loss_Scale", 1e-2)
                self.add_scalar("Total_Jerk_Loss", jerk_loss)
            else:
                jerk_loss = 0
            self.unsupervised_optim = self.minimizer("Unsupervised_Loss", force_loss + jerk_loss)
            # Supervised loss
            supervised_loss = l2_loss(seq[n//2].u[0] - seq[n//2].true)
            self.supervised_optim = self.minimizer("Supervised_Loss", supervised_loss)


    def step(self):
        if use_graph:
            self.tfstep(self.supervised_optim if self.value_supervised else self.unsupervised_optim)
        else:
            self.executor.set_dict(self.feed_dict(self.val_iterator, False))
            self.sequence.execute()

    def base_feed_dict(self):
        return {self.is_supervised: self.value_supervised }

    def action_write_forces(self):
        self.info("Computing forces...")
        baseline = self.view(self.baseline_force, all_batches=True)
        total = self.view(self.total_force, all_batches=True)
        gt = self.view(self.gt_force, all_batches=True)
        print("Baseline")
        print(baseline)
        print("Actual")
        print(total)
        print("Ground Truth")
        print(gt)
        np.savetxt(self.scene.subpath("forces_baseline.csv"), baseline, delimiter=", ")
        np.savetxt(self.scene.subpath("forces_%08d.csv" % self.time), total, delimiter=", ")
        np.savetxt(self.scene.subpath("forces_groundtruth.csv"), gt, delimiter=", ")
        self.info("Forces written to disc.")

    def action_plot_sequences(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        self.info("Computing frames...")
        pred_densities = [self.executor.lookup(self.sequence[i].u[-1]) for i in range(len(self.sequence))]
        true_densities = self.database.fixed_range("val", ["u%d" % i for i in range(n + 1)], range(self.validation_batch_size)).get_batch()
        if use_graph:
            pred_densities = self.view(pred_densities, all_batches=True)

        np.save(self.scene.subpath("sequence"), np.stack(pred_densities, 0))
        true_densities_ordered = [true_densities["u%d"%i]for i in range(len(true_densities))]
        np.save(self.scene.subpath("groundtruth"), np.stack(true_densities_ordered, 0))

        # for batch in range(pred_densities[0].shape[0]):
        #     batchdir = os.path.join(self.get_image_dir(), "sequence_%d"%batch)
        #     self.info("Plotting batch batch %d to %s" % (batch, batchdir))
        #     os.mkdir(batchdir)
        #     for i in range(len(true_densities)):
        #         true_density = true_densities["u%d"%i]
        #         pred_density = pred_densities[i]
        #
        #         plt.figure(figsize=(20, 10))
        #         # Ground Truth
        #         plt.subplot2grid((1, 3), (0, 0))
        #         plt.title('Ground Truth')
        #         plt.plot(true_density[batch,:,0])
        #         # Predicted
        #         plt.subplot2grid((1, 3), (0, 2))
        #         plt.title('Predicted')
        #         plt.plot(pred_density[batch,:,0])
        #         # Save file
        #         plt.savefig(os.path.join(batchdir, "It_%d_Sequence_%d.png" % (self.time, i)))
        #         plt.close()
        #
        # self.info("Saved all sequences to %s" % self.get_image_dir())
        return self

if "help" in sys.argv or "-help" in sys.argv or "--help" in sys.argv:
    print("First argument: n (integer)")
    print("Keywords: inference, preload, all_frames, adaptive, autorun, unsupervised, exact_force, forcenet")
    exit(0)
if len(sys.argv) >= 2:
    n = int(sys.argv[1])
else:
    n = 2
use_graph = True
preload = True
load_all_frames = "all_frames" in sys.argv
divide_strategy = "adaptive" if "adaptive" in sys.argv else "binary"
autorun = "autorun" in sys.argv
supervised = "supervised" in sys.argv
force_inference = "exact" if "exact_force" in sys.argv else "advection"
viscosity = 0.1

app = BurgerSM()
if autorun: app.play()

app.prepare()
app.action_write_forces()
app.action_plot_sequences()

app.show(display=("u (Predicted)", "u (Ground Truth)"), depth=31, production=__name__ != "__main__")