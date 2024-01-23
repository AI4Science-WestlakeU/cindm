# coding=utf-8
from __future__ import print_function
import sys
from phi.tf.model import *
from phi.control.control import *
from phi.control.iksm import *


class SmokeSM(TFModel):
    def __init__(self):
        TFModel.__init__(self, "SmokeSM %d"%n, "Slow motion density generation CNN training",
                         learning_rate=1e-3, data_validation_fraction=0.1,
                         training_batch_size=4, validation_batch_size=16,
                         model_scope_name="sm%d"%n)
        self.info("Mode: %s" % ("Graph" if use_graph else "Inference"))
        self.info("Preload: %s" % ("Yes" if preload else "No"))
        self.info("Load frames: %s" % ("All" if load_all_frames else "Start/End/Center"))
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
        # self.database.put_scenes(scenes("SmokeIK", "synthetic"), per_scene_indices=range(4, 4+n+1))
        if n <= 8:
            self.database.put_scenes(scenes("~/data/SmokeIK", "random_128x128"), per_scene_indices=range(4, 4+n*fac+1))
        else:
            self.database.put_scenes(scenes("~/data/SmokeIK", "random3p_128x128"), per_scene_indices=range(4, 4+n*fac+1))  # Something in this dataset causes NaN for small n
        self.info("Loaded %d scenes." % self.database.scene_count)

        self.finalize_setup([f.true for f in self.sequence if f.true is not None])

        # Load previously trained models
        models = {
            "IK": "~/model/smokeik/sim_000439/checkpoint_00009135",
            "SM2": "~/model/smokesm-2/sim_000147/checkpoint_00002483",
            "SM4": "~/model/smokesm-4/sim_000020/checkpoint_00001880",
            "SM8": "~/model/smokesm-8/sim_000010/checkpoint_00000953",
            "SM16": "~/model/smokesm-16/sim_000015/checkpoint_00005329",
            "SM32": "~/model/smokesm-32/sim_000008/checkpoint_00000249",
            "SM64": "~/model/smokesm-64/sim_000030/checkpoint_00130506"
        }
        # self.executor.load_checkpoints(n, models, preload_n=preload)

        self.executor.load_all_from(n, "/home/holl/model/smokeik/sim_000439/checkpoint_00009135",
                                    "/home/holl/model/refine-smokesm-32/sim_000002/checkpoint_00060966",
                                    lambda i: i)  # Unsupervised + Rough Joint SM refinement

        self.value_supervised = supervised

        self.display_time = EditableInt("Frame Display", n//2, (0, n))

        self.add_field("Density (Ground Truth)", lambda: self.view_batch("density%d"%self.display_time))
        self.add_field("Density (Real)", lambda: self.view(self.sequence[self.display_time].real))
        self.add_field("Density (Predicted)", lambda: self.view(self.sequence[self.display_time].pred[0]))
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
        self.sim = TFFluidSimulation([64 if half_res else 128] * 2, "open")

        # Placeholders
        if load_all_frames:
            true_densities = [self.sim.placeholder(name="density%d" % i) for i in range(n + 1)]
        else:
            true_densities = [ None ] * (n+1)
            for i in [0, n//2, n]:
                true_densities[i] = self.sim.placeholder(name="density%d" % i)

        if use_graph:
            self.executor = GraphBuilder(self.sim, true_densities, trainable_n=[n], info=self.info, force_inference=force_inference)
        else:
            self.executor = EagerExecutor(self.sim, true_densities, self.info, force_inference)

        seq = self.sequence = get_divide_strategy(divide_strategy)(n, self.executor)

        if use_graph:
            self.sequence.execute()

            # Density loss
            density_blur = 4.0
            final_density_loss = l2_loss(blur(seq[-1].real - seq[-1].true, density_blur)) * self.editable_float("Final_Density_Loss_Scale", 1e2)
            self.add_scalar("Final_Density_Loss_%.1f"%density_blur, final_density_loss)
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
            self.tfstep(self.supervised_optim if self.value_supervised else self.unsupervised_optim)
        else:
            self.executor.set_dict(self.feed_dict(self.val_iterator, False))
            self.sequence.execute()

    def base_feed_dict(self):
        return {self.is_supervised: self.value_supervised }


    def action_plot_sequences(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        self.info("Computing frames...")
        real_densities = [self.executor.lookup(self.sequence[i].real) for i in range(len(self.sequence))]
        pred_densities = [self.executor.lookup(self.sequence[i].pred[-1]) for i in range(len(self.sequence))]
        true_densities = self.database.fixed_range("val", ["density%d" % i for i in range(n + 1)], range(self.validation_batch_size)).get_batch()
        if use_graph:
            data = self.view(real_densities+pred_densities, all_batches=True)
            real_densities = data[:len(real_densities)]
            pred_densities = data[len(real_densities):]

        for batch in range(real_densities[0].shape[0]):
            batchdir = os.path.join(self.get_image_dir(), "sequence_%d"%batch)
            self.info("Plotting batch batch %d to %s" % (batch, batchdir))
            os.mkdir(batchdir)
            for i in range(len(true_densities)):
                true_density = true_densities["density%d"%i]
                real_density = real_densities[i]
                pred_density = pred_densities[i]

                plt.figure(figsize=(20, 10))
                # Ground Truth
                plt.subplot2grid((1, 3), (0, 0))
                plt.title('Ground Truth')
                plt.imshow(true_density[batch, :, :, 0], interpolation="nearest", cmap="bwr", origin="lower")
                # Real
                plt.subplot2grid((1, 3), (0, 1))
                plt.title('Real')
                plt.imshow(real_density[batch, :, :, 0], interpolation="nearest", cmap="bwr", origin="lower")
                # Predicted
                plt.subplot2grid((1, 3), (0, 2))
                plt.title('Predicted')
                plt.imshow(pred_density[batch, :, :, 0], interpolation="nearest", cmap="bwr", origin="lower")
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
preload = "preload" in sys.argv or not use_graph
load_all_frames = "all_frames" in sys.argv
divide_strategy = "adaptive" if "adaptive" in sys.argv else "binary"
autorun = "autorun" in sys.argv
supervised = "unsupervised" not in sys.argv
force_inference = "exact" if "exact_force" in sys.argv else "advection"
half_res = "half_res" in sys.argv
if "forcenet" in sys.argv: force_inference = "forcenet"

app = SmokeSM()
if autorun: app.play()
app.show(display=("Density (Predicted)", "Density (Ground Truth)"), depth=31, production=__name__ != "__main__")