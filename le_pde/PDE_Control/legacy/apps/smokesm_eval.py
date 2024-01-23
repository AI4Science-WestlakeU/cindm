# coding=utf-8
from __future__ import print_function
import sys
from phi.tf.model import *
from phi.control.control import *
from phi.control.iksm import *


class SmokeSM(TFModel):
    def __init__(self):
        TFModel.__init__(self, "Refine-SmokeSM %d"%n, "Slow motion density generation CNN training",
                         learning_rate=1e-5, data_validation_fraction=0.1,
                         training_batch_size=2, validation_batch_size=val_batch_size,
                         model_scope_name="sm%d"%n)
        self.info("Mode: %s" % ("Graph" if use_graph else "Inference"))
        self.info("Preload: %s" % ("Yes" if preload else "No"))
        self.info("Divide Strategy: %s" % divide_strategy)
        self.info("Force: %s" % force_inference)
        self.setup()

        self.info("Setting up database...")
        fac = 2 if half_res else 1
        for i in range(n+1):
            field = BatchSelect(lambda len, i=i: range(fac*i, len-fac*n+fac*i), "Density")
            if half_res:
                field = transform.Downsample(field)
            self.database.add("density%d"%i, augment.AxisFlip(2, field))

        # self.database.put_scenes(scenes('~/data/SmokeIK', 'random3p_128x128'), per_scene_indices=range(4, 4+n+1), logf=self.info)  # Something in this dataset causes NaN for small n
        self.database.put_scenes(scenes("~/data/control/simpleshapes"), per_scene_indices=range(n * fac + 1), logf=self.info)

        self.finalize_setup([f.true for f in self.sequence if f.true is not None])

        # Load previously trained models
        supervised_models = { # Learning rate 1e-3 decreased until 1e-5
            "IK": "~/model/smokeik/sim_000439/checkpoint_00009135",
            "SM2": "~/model/smokesm-2/sim_000236/checkpoint_00002643",
            "SM4": "~/model/smokesm-4/sim_000024/checkpoint_00002855",
            "SM8": "~/model/smokesm-8/sim_000013/checkpoint_00002041",
            "SM16": "~/model/smokesm-16/sim_000017/checkpoint_00035431",
            "SM32": "/home/holl/model/smokesm-32/sim_000014/checkpoint_00016361",
            "SM64": "/home/holl/model/smokesm-64/sim_000034/checkpoint_00002485",
        }

        models = {  # Supervised pretraining, then unsupervised (staggered)
            "IK": "~/model/smokeik/sim_000439/checkpoint_00009135",
            "SM2": "~/model/smokesm-2/sim_000147/checkpoint_00002483",
            "SM4": "~/model/smokesm-4/sim_000020/checkpoint_00001880",
            "SM8": "~/model/smokesm-8/sim_000010/checkpoint_00000953",
            "SM16": "~/model/smokesm-16/sim_000015/checkpoint_00005329",
            "SM32": "~/model/smokesm-32/sim_000008/checkpoint_00000249",
            "SM64": "~/model/smokesm-64/sim_000030/checkpoint_00130506"
        }
        # models = { # Joint training (staggered)
        #     "IK": "~/model/smokeik/sim_000439/checkpoint_00009135",
        #     "SM2":  "/home/holl/model/refine-smokesm-32/sim_000002/checkpoint_00060966",
        #     "SM4":  "/home/holl/model/refine-smokesm-32/sim_000002/checkpoint_00060966",
        #     "SM8":  "/home/holl/model/refine-smokesm-32/sim_000002/checkpoint_00060966",
        #     "SM16": "/home/holl/model/refine-smokesm-32/sim_000002/checkpoint_00060966",
        #     "SM32": "/home/holl/model/refine-smokesm-32/sim_000002/checkpoint_00060966",
        #     "SM64": "/home/holl/model/refine-smokesm-32/sim_000002/checkpoint_00060966"
        # }

        # self.executor.load_checkpoints(n, supervised_models, preload_n=preload)

        # def source_sm(i):
        #     if i == 64: return 32
        #     if i == 32: return 16
        #     return i
        # self.executor.load_all_from(n, "/home/holl/model/smokeik/sim_000439/checkpoint_00009135",
        #                             "/home/holl/model/refine-smokesm-32/sim_000002/checkpoint_00060966",
        #                             lambda i: i)  # staggered predictions

        # def source_sm(i):
        #     if i == 2: return 2
        #     else: return i // 2
        # def source_sm(i):
        #     if i > 32: return 32
        #     else: return i
        # self.executor.load_all_from(n, "/home/holl/model/smokeik/sim_000439/checkpoint_00009135",
        #                             "/home/holl/model/refine-smokesm-32/sim_000012/checkpoint_00001219",
        #                             lambda i: i)  # Refined predictions



        # Shapes

        # self.executor.load_all_from(n, "/home/holl/model/smokeik/sim_000439/checkpoint_00009135",
        #                             "/home/holl/model/refine-smokesm-32/sim_000004/checkpoint_00006811", 32) # Supervised squares

        # self.executor.load_all_from(n, "/home/holl/model/smokeik/sim_000439/checkpoint_00009135",
        #                             "/home/holl/model/refine-smokesm-16/sim_000001/checkpoint_00578845", lambda i: i) # Supervised + Rough Joint SM refinement

        # self.executor.load_all_from(n, "/home/holl/model/smokeik/sim_000439/checkpoint_00009135",
        #                             "/home/holl/model/refine-smokesm-16/sim_000009/checkpoint_00463391",
        #                             lambda i: i) # Supervised + Rough Joint SM refinement
        #
        # self.executor.load_all_from(n, "/home/holl/model/smokeik/sim_000439/checkpoint_00009135",
        #                             "/home/holl/model/refine-smokesm-16/sim_000012/checkpoint_00146094",
        #                             lambda i: i) # Supervised + 4px Joint SM refinement

        self.executor.load_all_from(n, "/home/holl/model/smokeik/sim_000439/checkpoint_00009135",
                                    "/home/holl/model/refine-smokesm-16/sim_000015/checkpoint_00286271",
                                    lambda i: i)  # SimpleShapes, Supervised + 4px + 2px Joint SM refinement binary

        # def extrapolate_shape_sms(i):
        #     if i == 64: return 16
        #     if i == 32: return 8
        #     if i == 16: return 4
        #     if i == 8: return 4
        #     if i == 4: return 2
        #     if i == 2: return 2
        #
        # self.executor.load_all_from(n, "/home/holl/model/smokeik/sim_000439/checkpoint_00009135",
        #                             "/home/holl/model/refine-smokesm-16/sim_000018/checkpoint_00219600",
        #                             lambda i: i)  # SimpleShapes, Interleaved



        self.value_supervised = supervised

        self.display_time = EditableInt("Frame Display", n//2, (0, n))

        self.add_field("Density (Ground Truth)", lambda: self.view_batch("density%d"%self.display_time))
        self.add_field("Density (Real)", lambda: self.view(self.sequence[self.display_time].real))
        self.add_field("Density (Predicted)", lambda: self.view(self.sequence[self.display_time].pred[0]))
        self.add_field("Velocity", lambda: self.view(self.sequence[self.display_time].velocity))
        self.add_field("Force", lambda: self.view(self.sequence[self.display_time].force))
        self.add_field("Jerk", lambda: self.view(self.sequence[self.display_time].jerk))
        if use_graph:
            self.add_field("Density L1", self.blurred_density_diff)

        if not use_graph:
            self.step()

    def setup(self):
        # Factors
        self.is_supervised = tf.placeholder(tf.float32, (), "is_supervised")
        self.add_scalar("Is_Supervised", self.is_supervised)
        # Simulation
        self.sim = TFFluidSimulation([64 if half_res else 128] * 2, "open")

        # Placeholders
        true_densities = [ None ] * (n+1)
        for i in [0, n]:
            true_densities[i] = self.sim.placeholder(name="density%d" % i)

        if use_graph:
            self.executor = GraphBuilder(self.sim, true_densities, trainable_n=range(n+1), info=self.info, force_inference=force_inference)
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
            final_density_loss = self.final_density_loss = l2_loss(self.blurred_density_diff, reduce_batches=False) * self.editable_float("Final_Density_Loss_Scale", 1e1) # 1e7 for 1/8 res 4px, 1e4 for 4px
            self.add_scalar("Final_Density_Loss", final_density_loss)
            # Force loss
            force_losses = []
            forces = []
            jerk_losses = []
            for frame in seq:
                if frame.force is not None:
                    force_losses.append(l2_loss(frame.force, reduce_batches=False))
                if frame.jerk is not None:
                    jerk_losses.append(l2_loss(frame.jerk, reduce_batches=False))
            force_loss = self.force_loss = tf.add_n(force_losses) * self.editable_float("Force_Loss_Scale", 1e-2)
            self.add_scalar("Total_Force_Loss", force_loss)
            self.total_force = math.add(forces)
            if jerk_losses:
                jerk_loss = tf.add_n(jerk_losses) * self.editable_float("Jerk_Loss_Scale", 1e-3)
                self.add_scalar("Total_Jerk_Loss", jerk_loss)
            else:
                jerk_loss = 0

    def validate(self, create_checkpoint=False):
        pass

    def step(self):
        if use_graph:
            pass
        else:
            self.executor.set_dict(self.feed_dict(self.val_iterator, False))
            self.sequence.execute()

    def base_feed_dict(self):
        return {self.is_supervised: self.value_supervised }


    def action_print_forces(self):
        self.info("Inferring density and velocity...")
        d, f = self.view([self.final_density_loss, self.force_loss], all_batches=True)
        self.info("Density: %f \pm %f"% (np.mean(d), np.std(d) / np.sqrt(d.shape[0])))
        self.info("Force: %f \pm %f"% (np.mean(f), np.std(f) / np.sqrt(f.shape[0])))


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
use_graph = True # "graph" in sys.argv
preload = True
val_batch_size = 100

divide_strategy = "adaptive" if "adaptive" in sys.argv else "binary"
supervised = "supervised" in sys.argv
force_inference = "exact" if "exact_force" in sys.argv else "advection"
half_res = "half_res" in sys.argv
if "forcenet" in sys.argv: force_inference = "forcenet"

app = SmokeSM()

app.prepare()
if use_graph:
    app.action_print_forces()
else:
    app.action_plot_sequences()

exit(0)

# app.show(display=("Density (Predicted)", "Density (Ground Truth)"), depth=31, production=__name__ != "__main__")