from phi.tf.model import *
from phi.tf.util import residual_block_1d
import phi.math as math


def sm_resnet(initial, target, frames, training=True, trainable=True, reuse=tf.AUTO_REUSE):
    frames = math.expand_dims(math.expand_dims(frames, -1), -1)
    frames = tf.tile(frames, [1]+list(initial.shape[1:]))
    y = tf.concat([initial, target, frames], axis=-1)
    downres_padding = sum([2**i for i in range(5)])
    y = tf.pad(y, [[0,0], [0,downres_padding], [0,0]])
    resolutions = [ y ]
    for i, filters in enumerate([4, 8, 16, 16, 16]):
        y = tf.layers.conv1d(resolutions[0], filters, 2, strides=2, activation=tf.nn.relu, padding='valid', name='downconv_%d'%i, trainable=trainable, reuse=reuse)
        for j in range(2):
            y = residual_block_1d(y, filters, name='downrb_%d_%d' % (i,j), training=training, trainable=trainable, reuse=reuse)
        resolutions.insert(0, y)

    for j, nb_channels in enumerate([16, 16, 16]):
        y = residual_block_1d(y, nb_channels, name='centerrb_%d' % j, training=training, trainable=trainable, reuse=reuse)

    for i, resolution_data in enumerate(resolutions[1:]):
        y = upsample2x(y)
        res_in = resolution_data[:, 0:y.shape[1], :]
        y = tf.concat([y, res_in], axis=-1)
        if i < len(resolutions)-2:
            y = tf.pad(y, [[0, 0], [0, 1], [0, 0]], mode='SYMMETRIC')
            y = tf.layers.conv1d(y, 16, 2, 1, activation=tf.nn.relu, padding='valid', name='upconv_%d' % i, trainable=trainable, reuse=reuse)
            for j, nb_channels in enumerate([16, 16]):
                y = residual_block_1d(y, nb_channels, 2, name='uprb_%d_%d' % (i, j), training=training, trainable=trainable, reuse=reuse)
        else:
            # Last iteration
            y = tf.pad(y, [[0,0], [0,1], [0,0]], mode='SYMMETRIC')
            y = tf.layers.conv1d(y, 1, 2, 1, activation=None, padding='valid', name='upconv_%d'%i, trainable=trainable, reuse=reuse)

    return y


def sim_step(u):
    u = advect(u, u)
    u += viscosity * laplace(u)
    return u


class BurgerCFE(TFModel):

    def __init__(self):
        TFModel.__init__(self, "Burger CFE", model_scope_name='CFE', data_validation_fraction=0.1)
        sim = self.sim = TFFluidSimulation([32], 'open', 100)

        u_true = [sim.placeholder(name="u%d" % i) for i in range(n+1)]
        u_reconstructed = self.u_reconstructed = [u_true[i] if i == 0 or i == n else None for i in range(n+1)]
        for i in range(1,n):
            with self.model_scope():
                u_reconstructed[i] = sm_resnet(u_reconstructed[i-1], u_true[-1], np.stack([(i-1)/32.0]*sim.batch_size).astype(np.float32))

        force_losses = []
        forces = []
        for i in range(n):
            force = u_reconstructed[i+1] - sim_step(u_reconstructed[i])
            force_losses.append(l2_loss(force))
            forces.append(l1_loss(force, reduce_batches=False))
        force_loss = math.add(force_losses) * self.editable_float("Force_Loss_Scale", 1e-1)
        self.minimizer("DiffPhys", force_loss)
        self.total_force = math.add(forces)

        for i in range(n+1):
            field = BatchSelect(lambda len, i=i: range(i, len-n+i), "u")
            self.database.add("u%d"%i, augment.AxisFlip(2, field))
        self.database.put_scenes(scenes("~/data/control/forced-burgerclash"), range(33), logf=self.info)
        self.finalize_setup(u_true)

        self.frame = EditableInt("Frame", 0, (0,n))

        self.add_field("Ground Truth", lambda: self.view_batch("u%d"%self.frame))
        self.add_field("Prediction", lambda: self.view(u_reconstructed[self.frame]))

    def action_write_to_disc(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        self.info("Computing...")
        total = self.view(self.total_force, all_batches=True)
        print(total)
        np.savetxt(self.scene.subpath("forces_%08d.csv" % self.time), total, delimiter=", ")

        u_reconstructed = self.view(self.u_reconstructed, all_batches=True)

        np.save(self.scene.subpath("sequence"), np.stack(u_reconstructed))
        self.info("Forces and sequence written to disc.")
        return self


viscosity = 0.1
n = 32
BurgerCFE().show(display=('Prediction', 'Ground Truth'))