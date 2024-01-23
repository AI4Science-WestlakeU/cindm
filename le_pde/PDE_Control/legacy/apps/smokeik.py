# coding=utf-8
from phi.data import augment, BatchSelect, scenes
from phi.tf.model import *
from phi.tf.util import *
from phi.control.control import blur
from phi.experimental import *
from phi.math.nd import spatial_sum


def ik_resnet(initial_density, target_density, training=False, trainable=True, reuse=None):
    y = tf.concat([initial_density, target_density], axis=-1)
    y = tf.pad(y, [[0,0], [0,1+2+4+4], [0,1+2+4+4], [0,0]])
    resolutions = [ y ]
    for i in range(1,4): # 1/2, 1/4, 1/8
        y = tf.layers.conv2d(resolutions[0], 16, 2, strides=2, activation=tf.nn.relu, padding="valid", name="downconv_%d"%i, trainable=trainable, reuse=reuse)
        for j, nb_channels in zip(range(3), [16, 16, 16]):
            y = residual_block(y, nb_channels, name="downrb_%d_%d" % (i,j), training=training, trainable=trainable, reuse=reuse)
        resolutions.insert(0, y)

    y = tf.layers.conv2d(y, 16, 2, 1, activation=tf.nn.relu, padding="valid", name="centerconv_1", trainable=trainable, reuse=reuse)
    for j, nb_channels in zip(range(3), [16, 16, 16]):
        y = residual_block(y, nb_channels, name="centerrb_%d" % j, training=training, trainable=trainable, reuse=reuse)

    for i in range(1, len(resolutions)):
        y = upsample2x(y)
        res_in = resolutions[i][:, 0:y.shape[1], 0:y.shape[2], :]
        y = tf.concat([y, res_in], axis=-1)
        if i < len(resolutions)-1:
            y = tf.pad(y, [[0, 0], [0, 1], [0, 1], [0, 0]], mode="SYMMETRIC")
            y = tf.layers.conv2d(y, 16, 2, 1, activation=tf.nn.relu, padding="valid", name="upconv_%d" % i, trainable=trainable, reuse=reuse)
            for j, nb_channels in zip(range(3), [16, 16, 16]):
                y = residual_block(y, nb_channels, 2, name="uprb_%d_%d" % (i, j), training=training, trainable=trainable, reuse=reuse)
        else:
            # Last iteration
            y = tf.pad(y, [[0,0], [0,1], [0,1], [0,0]], mode="SYMMETRIC")
            y = tf.layers.conv2d(y, 1, 2, 1, activation=None, padding="valid", name="upconv_%d"%i, trainable=trainable, reuse=reuse)

    return StaggeredGrid(y) # This is the vector potential


class SmokeIK(TFModel):
    def __init__(self):
        TFModel.__init__(self, "SmokeIK", "Solve for Velocity",
                         learning_rate=1e-3, data_validation_fraction=0.1,
                         training_batch_size=16, validation_batch_size=64,
                         model_scope_name="ik")
        self.info("Building graph...")
        self.setup()

        self.add_field("Density (Ground Truth)", lambda: self.view_batch("target_density") if self.value_view_advected else self.view_batch("initial_density"))
        self.add_field("Density (Model)", lambda: self.view(self.advected_density) if self.value_view_advected else self.view_batch("initial_density"))
        self.add_field("Velocity (Target)", lambda: self.view_batch("true_velocity"))
        self.add_field("Velocity (Model)", lambda: self.view(self.velocity))
        self.add_field(u"∇ Velocity (Model)", lambda: self.view(self.s_velocity_grad))
        self.add_field("Vector Potential (Model)", lambda: self.view(self.vec_pot))
        self.add_field(u"∇ Vector Potential (Model)", lambda: self.view(self.s_pot_grad))
        self.add_field("Density Residual (Model)", lambda: self.view_batch("target_density") - self.view(self.advected_density))
        self.add_field("Frame Difference (Ground Truth)", lambda: self.view_batch("target_density") - self.view_batch("initial_density"))
        self.add_field("Frame Difference (Model)", lambda: self.view(self.advected_density) - self.view_batch("initial_density"))
        self.add_field("Density Residual (Ground Truth)", lambda: self.view(self.true_advected_difference))
        self.add_field("Supervised Weight", lambda: self.view(self.supervised_weight_grad if self.value_gradient_weighting else self.supervised_weight_flat))
        self.add_field("Density (Frame 2, Ground Truth)", lambda: self.view(self.adv_true_density_2))
        self.add_field("Density (Frame 2, Model)", lambda: self.view(self.adv_model_density_2))

        self.info("Setting up database...")
        # AugmentFlip needs special handling for vector fields
        self.database.add("initial_density", augment.AxisFlip(2, BatchSelect(lambda len: range(len - 1), "Density")))
        self.database.add("target_density", augment.AxisFlip(2, BatchSelect(lambda len: range(1, len), "Density")))
        self.database.add("true_velocity", augment.AxisFlip(2, BatchSelect(lambda len: range(len - 1), "Velocity")))
        self.database.put_scenes(scenes("SmokeIK", "random_128x128"), per_scene_indices=range(4, 6))
        self.info("Loaded %d scenes." % self.database.scene_count)

        self.finalize_setup([self.initial_density, self.target_density, self.true_velocity])
        self.value_supervised = True
        self.value_gradient_weighting = True
        self.value_regularization_factor = 1e-6

        self.value_view_advected = True

    def step(self):
        self.tfstep((self.supervised_optim_grad if self.value_gradient_weighting else self.supervised_optim_flat)
                    if self.value_supervised else self.unsupervised_optim)

    def base_feed_dict(self):
        return {self.is_supervised: self.value_supervised,
                self.regularization_factor: self.value_regularization_factor }
    
    def setup(self):
        self.sim = TFFluidSimulation([128] * 2, "open")

        self.initial_density = self.sim.placeholder(name="initial_density")
        self.target_density = self.sim.placeholder(name="target_density")
        self.true_velocity = self.sim.placeholder("staggered", name="true_velocity")

        with self.model_scope():
            self.vec_pot = ik_resnet(self.initial_density, self.target_density, training=self.training)
        with tf.variable_scope("curl"):
            self.velocity = self.vec_pot.curl()
        self.velocity = self.velocity.pad(0, 1, "symmetric")
        self.advected_density = self.velocity.advect(self.initial_density)

        # Supervised Loss
        self.supervised_weight_flat = StaggeredGrid.from_scalar(self.target_density + self.initial_density, (1, 1)).normalize()
        self.supervised_weight_grad = (StaggeredGrid.gradient(self.target_density).abs() + StaggeredGrid.gradient(self.initial_density).abs()).soft_sqrt().normalize()
        self.supervised_loss_flat = l2_loss((self.velocity - self.true_velocity).batch_div(self.true_velocity.total()) * self.supervised_weight_flat) * 1e5
        self.supervised_loss_grad = l2_loss((self.velocity - self.true_velocity).batch_div(self.true_velocity.total()) * self.supervised_weight_grad) * 1e5
        # Unsupervised Loss
        adv_true_velocity = self.true_velocity.advect(self.true_velocity)
        adv_model_velocity = self.velocity.advect(self.velocity)
        self.adv_true_density_2 = adv_true_velocity.advect(self.target_density)
        self.adv_model_density_2 = adv_model_velocity.advect(self.advected_density)
        density_loss_1 = l2_loss(blur(self.advected_density - self.target_density, 2.0, cutoff=3) / spatial_sum(self.target_density)) * 1e6
        density_loss_2 = l2_loss(blur(self.adv_model_density_2 - self.adv_true_density_2, 2.0, cutoff=3) / spatial_sum(self.target_density)) * 1e6
        self.unsupervised_loss = density_loss_1 + density_loss_2
        # Regularization
        self.regularization_factor = tf.placeholder(tf.float32, (), "regularization_factor")
        self.add_scalar("Regularization_Factor", self.regularization_factor)
        self.reg_loss = l1_loss(self.velocity) * self.regularization_factor

        # self.supervised_optim_flat = self.minimizer("Supervised_Loss_Flat", self.supervised_loss_flat, reg=self.reg_loss)
        self.supervised_optim_grad = self.minimizer("Supervised_Loss_Grad", self.supervised_loss_grad, reg=self.reg_loss)
        self.unsupervised_optim = self.minimizer("Unsupervised_Loss", self.unsupervised_loss, reg=self.reg_loss)

        self.s_velocity_grad, self.s_pot_grad = tf.gradients(self.unsupervised_loss, [self.velocity.staggered, self.vec_pot.staggered])
        # self.u_velocity_grad, self.u_pot_grad = tf.gradients(self.supervised_loss_grad, [self.velocity.staggered, self.vec_pot.staggered])

        self.is_supervised = tf.placeholder(tf.float32, (), "is_supervised")
        self.add_scalar("Is_Supervised", self.is_supervised)
        self.add_scalar("Density_1", density_loss_1)
        self.add_scalar("Density_2", density_loss_2)

        self.true_advected_difference = self.target_density - self.sim.conserve_mass(self.true_velocity.advect(self.initial_density), self.initial_density)


app = SmokeIK().show(display=("Velocity (Model)", "Velocity (Target)"), depth=31, production=__name__!="__main__")