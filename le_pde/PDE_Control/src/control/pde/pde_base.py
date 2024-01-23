from phi.tf.flow import struct, isplaceholder, consecutive_frames, StateCollection, placeholder, State
import tensorflow as tf
if tf.__version__[0] == '2':
    tf = tf.compat.v1
    tf.disable_eager_execution()


class PDE(object):

    def __init__(self):
        """
The constructor can be used to specify properties of the physical system shared by all examples.
Additionally, variables that are initialized in `create_pde` can be set to `None` in the constructor.
        """
        self.fields = {}
        self.scalars = {}

    def create_pde(self, world, control_trainable, constant_prediction_offset):
        """
Creates all states and physics objects and adds them to the given world.
This method should not call `world.step()` but simply specify the physical system.
The state of the world after this method call is not used in the simulation. It should, however, contain tensors of the correct shapes and types.

External influences like control forces should also be added to the world as states with accompanying physics objects.

Physics objects depending on the prediction of the next state, can add the following dependency:
`StateDependency("next_state_prediction", "next_state_prediction", single_state=True, blocking=True)`
This will result in a `NextStatePrediction` object being passed to `Physics.step(state, dt, next_state_prediction)`.
The predicted world state can be accessed as `next_state_prediction.prediction`.

Example:
    A neural network should predict a force that influences the velocity of the simulation each step. The force should depend on the previous state of the simulation.
Solution:
    Add a `FieldEffect` state, initially zero. Implement the corresponding `Physics` object to depend on the observable quantities of the simulation, run the network.
    Make sure the `StateDepencencies` are blocking to ensure the force field is predicted before the simulation is executed.
    In `Physics.step()`, run the nerual network and store the predicted force field in the `FieldEffect`.
        :param world: world that all states and physics should be added to
        :param control_trainable: whether the CFE network should be trainable, i.e. adjusted during optimization
        :param constant_prediction_offset: (not used)
        """
        raise NotImplementedError(self)

    def placeholder_state(self, world, age):
        """
Creates a TensorFlow placeholder world state for the simulation at time `age`.
The default implementation creates placeholders for all data-holding variables in the world.

The world should not be altered in any way by this method.
        :param world: world that was previously initialized via `create_pde`
        :param age: the simulation time of the state, 0 for the initial state, n*dt for the nth state.
        :return: world state at time `age`, holding placeholders for variable data
        :rtype: StateCollection
        """
        with struct.VARIABLES:
            with struct.DATA:
                placeholders = placeholder(world.state.staticshape)
        result = struct.map_item(State.age, lambda _: age, placeholders)
        return result

    def predict(self, n, initial_worldstate, target_worldstate, trainable):
        """
Predicts an intermediate world state given initial and target state.
This method can be implemented via an observation prediction (OP) model.

Note that the predicted world state will not be passed to the solver.
Instead, it can be retrieved via the `NextStatePrediction` state when the center state is computed by the simulation.
See the documentation of `create_pde`.

        :param n: number of frames betweeen initial_worldstate and target_worldstate
        :param initial_worldstate: initial world state
        :param target_worldstate: target world state
        :return: predicted world state exactly between initial_worldstate and target_worldstate
        """
        raise NotImplementedError(self)

    def target_matching_loss(self, target_state, actual_state):
        """
Computes the observation loss function, comparing `target_state` and `actual_state`.
This loss can be used both as a supervised loss and as a differentiable physics loss.

Returns None if no such loss exists.
        :param target_state: target world state
        :type target_state: StateCollection
        :param actual_state: world state simulated with current parameters
        :type actual_state: StateCollection
        :return: scalar TensorFlow tensor, e.g. math.l2_loss(target_state.mystate.myvalue - actual_state.mystate.myvalue)
        """
        raise NotImplementedError(self)

    def total_force_loss(self, states):
        """
Evaluates the total force exerted on the system that follows the trajectory described by `states`.
If the force is stored explicitly in the world state, this method can simply compute the sum.

Returns None if no such loss exists.
        :param states: ordered list of world states
        :return: scalar TensorFlow tensor
        """
        raise NotImplementedError(self)


def property_name(trace): 
    return trace.name


def collect_placeholders_channels(placeholder_states, trace_to_channel=property_name):
    if trace_to_channel is None:
        trace_to_channel = property_name
    placeholders = []
    channels = []

    for i, state in enumerate(placeholder_states):
        if state is not None:
            traces = struct.flatten(state, trace=True)
            for trace in traces:
                if isplaceholder(trace.value):
                    placeholders.append(trace.value)
                    channel = trace_to_channel(trace)
                    channels.append(consecutive_frames(channel, len(placeholder_states))[i])
    return placeholders, tuple(channels)
