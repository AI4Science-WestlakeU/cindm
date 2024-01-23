# coding=utf-8
from phi.data import augment, BatchSelect, scenes
from phi.tf.model import *
from phi.tf.util import *
from phi.control.control import blur
from phi.experimental import *
from phi.math.nd import spatial_sum
from phi import math
from phi.solver.cuda.cuda import CudaPressureSolver
from phi.solver.sparse import SparseCGPressureSolver

# DenseNet supervised    /home/holl/model/indirect-smokeik/sim_000053/checkpoint_00002315
# DenseNet unsupervised  /home/holl/model/indirect-smokeik/sim_000053/checkpoint_00002836

# IKResnet supervised    /home/holl/model/indirect-smokeik/sim_000054/checkpoint_00001125
# IKResnet unsupervised  /home/holl/model/indirect-smokeik/sim_000054/checkpoint_00002700

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


class SmokeIK(TFModel):
    def __init__(self):
        TFModel.__init__(self, 'Indirect SmokeIK',
                         learning_rate=1e-3, data_validation_fraction=0.1,
                         training_batch_size=16, validation_batch_size=16,
                         model_scope_name='ik', stride=10)
        self.info('Building graph...')

        self.sim = TFFluidSimulation([128] * 2, DomainBoundary([(False, True), (False, False)]), force_use_masks=True, batch_size=16)

        self.initial_density = self.sim.placeholder(name='initial_density')
        self.target_density = self.sim.placeholder(name='target_density')
        self.true_velocity = self.sim.placeholder('staggered', name='true_velocity')

        with self.model_scope():
            optimizable_velocity = ik_resnet(self.initial_density, self.target_density, self.training)
        optimizable_velocity = optimizable_velocity.pad(0, 1, 'symmetric').staggered
        control_mask = self.sim.ones("staggered")
        control_mask.staggered[:, 5:, 20:-20, :] = 0
        self.divergent_velocity = optimizable_velocity * control_mask
        self.velocity = self.sim.divergence_free(self.divergent_velocity, solver=SparseCGPressureSolver(), accuracy=1e-5)
        self.velocity = self.sim.with_boundary_conditions(self.velocity)
        self.advected_density = self.velocity.advect(self.initial_density)

        # Supervised Loss
        self.supervised_weight_flat = StaggeredGrid.from_scalar(self.target_density + self.initial_density, (1, 1))
        self.supervised_loss_flat = l2_loss((self.velocity - self.true_velocity) * self.supervised_weight_flat) * 1e-1
        reg = l1_loss(self.divergent_velocity.divergence()) * self.editable_float("Regularization_Factor", 1e-4)
        self.supervised_optim_flat = self.minimizer("Supervised_Loss_Flat", self.supervised_loss_flat, reg=reg)
        self.weighted_target = self.supervised_weight_flat * self.true_velocity

        # Unsupervised Loss
        adv_true_velocity = self.true_velocity.advect(self.true_velocity)
        adv_model_velocity = self.velocity.advect(self.velocity)
        self.adv_true_density_2 = adv_true_velocity.advect(self.target_density)
        self.adv_model_density_2 = adv_model_velocity.advect(self.advected_density)
        density_loss_1 = l2_loss(
            blur(self.advected_density - self.target_density, 2.0, cutoff=3) / spatial_sum(self.target_density)) * 1e6
        density_loss_2 = l2_loss(blur(self.adv_model_density_2 - self.adv_true_density_2, 2.0, cutoff=3) / spatial_sum(
            self.target_density)) * 1e6
        self.unsupervised_loss = (density_loss_1 + density_loss_2) * self.editable_float("Unsupervised_Loss_Scale", 1e-2)
        # Regularization
        self.unsupervised_optim = self.minimizer('Unsupervised_Loss', self.unsupervised_loss)

        self.add_scalar('Density_1', density_loss_1)
        self.add_scalar('Density_2', density_loss_2)

        self.info('Setting up database...')
        # AugmentFlip needs special handling for vector fields
        self.database.add('initial_density', BatchSelect(lambda len: range(len - 1), 'Density'))
        self.database.add('target_density', BatchSelect(lambda len: range(1, len), 'Density'))
        self.database.add('true_velocity', BatchSelect(lambda len: range(len - 1), 'Velocity'))
        self.database.put_scenes(scenes('~/data/control/bounded-squares'), per_scene_indices=range(2), logf=self.info)
        self.finalize_setup([self.initial_density, self.target_density, self.true_velocity])
        build_obstacles(self.sim)

        self.value_view_advected = True
        self.value_supervised = True
        self.is_supervised = tf.placeholder(tf.float32, (), "is_supervised")
        self.add_scalar("Is_Supervised", self.is_supervised)

        self.add_field('Density (Ground Truth)',
                       lambda: self.view_batch('target_density') if self.value_view_advected else self.view_batch(
                           'initial_density'))
        self.add_field('Density (Model)',
                       lambda: self.view(self.advected_density) if self.value_view_advected else self.view_batch(
                           'initial_density'))
        self.add_field('Velocity (Target)', lambda: self.view_batch('true_velocity'))
        self.add_field('Velocity (Weighted Target)', self.weighted_target)
        self.add_field('Velocity (Model, before solve)', self.divergent_velocity.staggered)
        self.add_field('Velocity (Model)', lambda: self.view(self.velocity.staggered))
        self.add_field('Density Residual (Model)',
                       lambda: self.view_batch('target_density') - self.view(self.advected_density))
        self.add_field('Frame Difference (Ground Truth)',
                       lambda: self.view_batch('target_density') - self.view_batch('initial_density'))
        self.add_field('Frame Difference (Model)',
                       lambda: self.view(self.advected_density) - self.view_batch('initial_density'))
        self.add_field('Density (Frame 2, Ground Truth)', self.adv_true_density_2)
        self.add_field('Density (Frame 2, Model)', self.adv_model_density_2)
        self.add_field('Domain', self.sim.extended_fluid_mask)
        self.add_field('Domain (Velocity)', lambda: self.view(self.sim._velocity_mask.staggered))

    def step(self):
        self.tfstep(self.supervised_optim_flat if self.value_supervised else self.unsupervised_optim)

    def base_feed_dict(self):
        return {self.is_supervised: self.value_supervised }


app = SmokeIK().show(display=('Velocity (Model)', 'Density (Ground Truth)'), production=__name__ != '__main__')
