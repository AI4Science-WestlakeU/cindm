# coding=utf-8
from __future__ import print_function
import sys
from phi.tf.model import *
from phi.control.control import *
from phi.control.iksm import *


class SmokeSM(TFModel):
    def __init__(self):
        TFModel.__init__(self, "Multishape-SmokeSM %d"%n, "Slow motion density generation CNN training",
                         learning_rate=1e-3, data_validation_fraction=1.0,
                         training_batch_size=2, validation_batch_size=2,
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
        self.database.put_scenes(scenes("~/data/control/specificshapes/i1"), per_scene_indices=range(n+1), dataset="val")
        self.database.put_scenes(scenes("~/data/control/specificshapes/i1"), per_scene_indices=range(n+1), dataset="train")
        self.finalize_setup([f.true for f in self.sequence if f.true is not None])

        # Load previously trained models
        sm_refined = "/home/holl/model/refine-smokesm-16/sim_000018/checkpoint_00219600"
        sm_staggered = "/home/holl/model/refine-smokesm-16/sim_000015/checkpoint_00286271"
        self.executor.load_all_from(n, "/home/holl/model/smokeik/sim_000439/checkpoint_00009135",
                                    sm_refined,
                                    lambda i: i)

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

        self.executor = MultiShapeEagerExecutor(self.sim, true_densities, self.info, force_inference)
        self.sequence = get_divide_strategy(divide_strategy)(n, self.executor)


    def step(self):
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
        real_velocities = [self.executor.lookup(self.sequence[i].velocity.staggered) for i in range(len(self.sequence) - 1)]
        real_vecpots = [self.executor.lookup(self.sequence[i].vec_pot.staggered) for i in range(len(self.sequence) - 1)]
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
        np.save(self.scene.subpath("seq_vecpot"), np.stack(real_vecpots, 1))


        for batch in [0,1]:  # range(real_densities[0].shape[0]):
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
use_graph = False
preload = "preload" in sys.argv or not use_graph
load_all_frames = "all_frames" in sys.argv
divide_strategy = "adaptive" if "adaptive" in sys.argv else "binary"
autorun = "autorun" in sys.argv
supervised = "unsupervised" not in sys.argv
force_inference = "exact" if "exact_force" in sys.argv else "advection"
half_res = "half_res" in sys.argv
if "forcenet" in sys.argv: force_inference = "forcenet"

app = SmokeSM()
app.prepare().action_plot_sequences()
# if autorun: app.play()
# app.show(display=("Density (Predicted)", "Density (Ground Truth)"), depth=31, production=__name__ != "__main__", port=8050)