from phi.tf.flow import tf, np, StateDependency, Physics, Burgers, BurgersVelocity, math, struct, AnalyticField, residual_block_1d
from .pde_base import PDE


class BurgersPDE(PDE):

    def __init__(self, domain, viscosity, dt):
        PDE.__init__(self)
        self.domain = domain
        self.viscosity = viscosity
        self.dt = dt

    def create_pde(self, world, control_trainable, constant_prediction_offset):
        world.reset(world.batch_size, add_default_objects=False)
        u0 = BurgersVelocity(self.domain, viscosity=self.viscosity, batch_size=world.batch_size, name='burgers')
        world.add(u0, ReplacePhysics())

    def target_matching_loss(self, target_state, actual_state):
        # Only needed for supervised initialization
        diff = target_state.burgers.velocity.data - actual_state.burgers.velocity.data
        loss = math.l2_loss(diff)
        return loss

    def total_force_loss(self, states):
        l2 = []
        l1 = []
        for s1, s2 in zip(states[:-1], states[1:]):
            natural_evolution = Burgers().step(s1.burgers, dt=self.dt)
            diff = s2.burgers.velocity - natural_evolution.velocity
            l2.append(math.l2_loss(diff.data))
            l1.append(math.l1_loss(diff.data))
        l2 = math.sum(l2)
        l1 = math.sum(l1)
        self.scalars["Total Force"] = l1
        return l2

    def predict(self, n, initial_worldstate, target_worldstate, trainable):
        b1, b2 = initial_worldstate.burgers, target_worldstate.burgers
        with tf.variable_scope("OP%d" % n):
            predicted_tensor = op_resnet(b1.velocity.data, b2.velocity.data)
        new_field = b1.copied_with(velocity=predicted_tensor, age=(b1.age + b2.age) / 2.)
        return initial_worldstate.state_replaced(new_field)


def op_resnet(initial, target, training=True, trainable=True, reuse=tf.AUTO_REUSE):
    # Set up Tensor y
    y = tf.concat([initial, target], axis=-1)
    downres_padding = sum([2 ** i for i in range(5)])  # 1+2+4+8+16=31
    y = tf.pad(y, [[0, 0], [0, downres_padding], [0, 0]], mode="CONSTANT", constant_values=0)
    resolutions = [y]
    # Add 1D convolution layers with varying kernel sizes:  1x conv1d(y, kernel), 2x residual block (2x con1d 2x ReLu)
    for i, filters in enumerate([4, 8, 16, 16, 16]):
        y = tf.layers.conv1d(
            resolutions[0], filters, kernel_size=2, strides=2, activation=tf.nn.relu, padding="valid", name="downconv_%d" % i, trainable=trainable, reuse=reuse
        )
        for j in range(2):
            y = residual_block_1d(y, filters, name="downrb_%d_%d" % (i, j), training=training, trainable=trainable, reuse=reuse)
        resolutions.insert(0, y)
    # Add 1D convolution layers with equal kernel size:  1x conv1d(y, kernel)
    for j, nb_channels in enumerate([16, 16, 16]):
        y = residual_block_1d(y, nb_channels, name="centerrb_%d" % j, training=training, trainable=trainable, reuse=reuse)
    # Add
    for i, resolution_data in enumerate(resolutions[1:]):
        y = math.upsample2x(y)
        res_in = resolution_data[:, 0:y.shape[1], :]
        y = tf.concat([y, res_in], axis=-1)
        if i < len(resolutions) - 2:
            y = tf.pad(tensor=y, paddings=[[0, 0], [0, 1], [0, 0]], mode="SYMMETRIC")
            y = tf.layers.conv1d(
                y, filters=16, kernel_size=2, strides=1, activation=tf.nn.relu, padding="valid", name="upconv_%d" % i, trainable=trainable, reuse=reuse
            )
            for j, nb_channels in enumerate([16, 16]):
                y = residual_block_1d(y, nb_channels, 2, name="uprb_%d_%d" % (i, j), training=training, trainable=trainable, reuse=reuse)
        else:
            # Last iteration
            y = tf.pad(tensor=y, paddings=[[0, 0], [0, 1], [0, 0]], mode="SYMMETRIC")
            y = tf.layers.conv1d(
                y, filters=1, kernel_size=2, strides=1, activation=None, padding="valid", name="upconv_%d" % i, trainable=trainable, reuse=reuse
            )
    return y


class ReplacePhysics(Physics):

    def __init__(self):
        Physics.__init__(self, dependencies=[StateDependency("next_state_prediction", "next_state_prediction", single_state=True, blocking=True)])

    def step(self, state, dt=1.0, next_state_prediction=None):
        return next_state_prediction.prediction.burgers


@struct.definition()
class GaussianClash(AnalyticField):

    def __init__(self, batch_size):
        AnalyticField.__init__(self, rank=1)
        self.batch_size = batch_size

    def sample_at(self, idx, collapse_dimensions=True):
        leftloc = np.random.uniform(0.2, 0.4, self.batch_size)
        leftamp = np.random.uniform(0, 3, self.batch_size)
        leftsig = np.random.uniform(0.05, 0.15, self.batch_size)
        rightloc = np.random.uniform(0.6, 0.8, self.batch_size)
        rightamp = np.random.uniform(-3, 0, self.batch_size)
        rightsig = np.random.uniform(0.05, 0.15, self.batch_size)
        idx = np.swapaxes(idx, 0, -1)  # batch last to match random values
        left = leftamp * np.exp(-0.5 * (idx - leftloc) ** 2 / leftsig ** 2)
        right = rightamp * np.exp(-0.5 * (idx - rightloc) ** 2 / rightsig ** 2)
        result = left + right
        result = np.swapaxes(result, 0, -1)
        return result

    @struct.constant()
    def data(self, data):
        return data


@struct.definition()
class GaussianForce(AnalyticField):
    def __init__(self, batch_size):
        AnalyticField.__init__(self, rank=1)
        self.loc = np.random.uniform(0.4, 0.6, batch_size)
        self.amp = np.random.uniform(-0.05, 0.05, batch_size) * 32
        self.sig = np.random.uniform(0.1, 0.4, batch_size)

    def sample_at(self, idx, collapse_dimensions=True):
        idx = np.swapaxes(idx, 0, -1)  # batch last to match random values
        result = self.amp * np.exp(-0.5 * (idx - self.loc) ** 2 / self.sig ** 2)
        result = np.swapaxes(result, 0, -1)
        return result

    @struct.constant()
    def data(self, data):
        return data
