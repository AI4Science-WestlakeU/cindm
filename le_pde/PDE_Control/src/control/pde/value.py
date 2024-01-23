from phi.tf.flow import StateDependency, Physics, ConstantField, FieldEffect, FieldPhysics, ADD
from .pde_base import PDE


class ScalarEffectControl(Physics):

    def __init__(self):
        Physics.__init__(self, [StateDependency('scalar', 'scalar', single_state=True),
                                StateDependency('pred', 'next_state_prediction', single_state=True, blocking=True)])

    def step(self, effect, dt=1.0, scalar=None, pred=None):
        force = pred.prediction.scalar.data - scalar.data
        return effect.copied_with(field=effect.field.with_data(force), age=effect.age + dt)


class IncrementPDE(PDE):

    def create_pde(self, world, control_trainable, constant_prediction_offset):
        world.reset(world.batch_size, add_default_objects=False)
        world.add(ConstantField(0.0, name='scalar', flags=()), physics=FieldPhysics('scalar'))
        world.add(FieldEffect(ConstantField(0, flags=()), ['scalar'], ADD), physics=ScalarEffectControl())

    def target_matching_loss(self, target_state, actual_state):
        return None

    def total_force_loss(self, states):
        return None

    def predict(self, n, initial, target, trainable):
        center_age = (initial.scalar.age + target.scalar.age) / 2
        new_field = initial.scalar.copied_with(data=(initial.scalar.data + target.scalar.data) * 0.5, flags=(), age=center_age)
        return initial.state_replaced(new_field)
