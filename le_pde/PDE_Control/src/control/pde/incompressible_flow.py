from phi.tf.flow import tf, psquare, math, Physics, StateDependency, staggered_curl_2d, CenteredGrid, advect
from phi.tf.standard_networks import u_net
from .pde_base import PDE


class IncompressibleFluidPDE(PDE):

    def __init__(self, domain, dt):
        PDE.__init__(self)
        self.domain = domain
        self.dt = dt

    def create_pde(self, world, control_trainable, constant_prediction_offset):
        world.reset(world.batch_size, add_default_objects=True)
        world.add(CenteredGrid.sample(0, self.domain, batch_size=world.batch_size, name='density'), physics=ControlledIncompressibleFlow(self.domain, control_trainable))

    def predict(self, n, initial_worldstate, target_worldstate, trainable):
        d1, d2 = initial_worldstate.density, target_worldstate.density
        with tf.variable_scope('OP%d' % n):
            density_pred = u_net(self.domain, [d1, d2], d1, levels=3, trainable=trainable, reuse=tf.AUTO_REUSE).copied_with(age=(d1.age + d2.age) / 2.).copied_with(name=d1.name)
        return initial_worldstate.state_replaced(density_pred)

    def target_matching_loss(self, target_state, actual_state):
        diff = target_state.density.data - actual_state.density.data
        diff_fft = psquare(math.fft(diff))
        k = math.fftfreq(diff.shape[1:-1])
        gauss_weights = math.exp(-0.5 * k ** 2 / 0.01 ** 2)
        loss = math.l1_loss(diff_fft * gauss_weights)
        return loss

    def total_force_loss(self, states):
        loss = 0
        count = 0
        for state1, state2 in zip(states[:-1], states[1:]):
            if hasattr(state1.density, 'velocity'):
                v1, v2 = state1.density.velocity, state2.density.velocity
                adv_vel = advect.semi_lagrangian(v1, v1, self.dt)
                loss += math.l2_loss(adv_vel - v2)
                count += 1
        if count == 0:
            return None
        else:
            return loss / count


class ControlledIncompressibleFlow(Physics):

    def __init__(self, domain, trainable):
        Physics.__init__(self, dependencies=[StateDependency('pred', 'next_state_prediction', single_state=True, blocking=True)])
        self.domain = domain
        self.trainable = trainable

    def step(self, density, dt=1.0, pred=None):
        with tf.variable_scope('CFE'):
            velocity_potential = u_net(self.domain, [density, pred.prediction.density], density, levels=2, trainable=self.trainable, reuse=tf.AUTO_REUSE)
        velocity = staggered_curl_2d(velocity_potential)
        next_density = advect.semi_lagrangian(density, velocity, dt).copied_with(age=density.age + dt)
        next_density.velocity = velocity
        return next_density
