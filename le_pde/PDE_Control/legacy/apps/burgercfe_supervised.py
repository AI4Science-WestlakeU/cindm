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


class BurgerCFE(TFModel):

    def __init__(self):
        TFModel.__init__(self, 'Supervised Burger CFE', model_scope_name='CFE', data_validation_fraction=0.1)
        sim = self.sim = TFFluidSimulation([32], 'open', 100)

        initial = sim.placeholder(name='initial')
        next = sim.placeholder(name='next')
        frame = tf.placeholder(tf.float32, (sim.batch_size,), 'frame')
        target = sim.placeholder(name='target')
        with self.model_scope():
            prediction = sm_resnet(initial, target, frame / 32.0)
        loss = l2_loss(prediction - next)    
        self.minimizer('Supervised_Loss', loss)

        self.database.add('initial', BatchSelect(lambda len: range(len-1), 'u'))  # TODO last frame doesn't need to be reconstructed
        self.database.add('next', BatchSelect(lambda len: range(1, len), 'u'))
        self.database.add('target', BatchSelect(lambda len: [len-1]*len, 'u'))
        self.database.add('frame', Frame())
        self.database.put_scenes(scenes('~/data/control/forced-burgerclash'), range(33), logf=self.info)
        self.finalize_setup([initial, next, target, frame])

        self.add_field('Initial', 'initial')
        self.add_field('Next', 'next')
        self.add_field('Target', 'target')
        self.add_field('Frame', 'frame')
        self.add_field('Prediction', prediction)


n = 32
BurgerCFE().show(display=('Prediction', 'Next'))
