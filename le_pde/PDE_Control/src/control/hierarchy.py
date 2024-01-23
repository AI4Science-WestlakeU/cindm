import os
from phi.flow import StateProxy, State, struct, StateCollection
from .sequences import SeqFrame, PartitioningExecutor, TYPE_KEYFRAME, TYPE_UNKNOWN, LinearSequence


class StateFrame(SeqFrame):

    def __init__(self, index, type):
        SeqFrame.__init__(self, index, type)
        self.worldstate = None

    def __getitem__(self, item):
        if isinstance(item, StateProxy):
            item = item.state
        return self.worldstate[item]


class PDEExecutor(PartitioningExecutor):

    def __init__(self, world, pde, target_state, trainable_networks, dt):
        self.world = world
        self.pde = pde
        self.worldsteps = 0
        self.next_state_prediction = NextStatePrediction(None)
        self.world.add(self.next_state_prediction)
        self.target_state = target_state
        self.trainable_networks = trainable_networks
        self.dt = dt

    def create_frame(self, index, step_count):
        frame = StateFrame(index, type=TYPE_KEYFRAME if index == 0 or index == step_count else TYPE_UNKNOWN)
        if index == 0:
            frame.worldstate = self.world.state
        if index == step_count:
            frame.worldstate = self.target_state
        return frame

    def execute_step(self, initial_frame, target_frame, sequence):
        PartitioningExecutor.execute_step(self, initial_frame, target_frame, sequence)
        assert initial_frame.index == self.worldsteps == target_frame.index - 1
        ws = initial_frame.worldstate
        if isinstance(sequence, LinearSequence):
            predicted_ws = self.target_state
        else:
            assert target_frame.worldstate is not None
            predicted_ws = target_frame.worldstate
        target_pred = ws[self.next_state_prediction].copied_with(prediction=predicted_ws)
        initial_frame.worldstate = ws.state_replaced(target_pred)
        self.world.state = initial_frame.worldstate
        self.world.step(dt=self.dt)
        self.worldsteps += 1
        if target_frame is sequence[-1]:
            self.world.remove(NextStatePrediction)
        target_frame.worldstate = self.world.state

    def partition(self, n, initial_frame, target_frame, center_frame):
        PartitioningExecutor.partition(self, n, initial_frame, target_frame, center_frame)
        center_frame.worldstate = self.pde.predict(n, initial_frame.worldstate, target_frame.worldstate, trainable='OP%d' % n in self.trainable_networks)

        if center_frame.index == self.worldsteps + 1:
            assert center_frame.worldstate is not None
            old_state = self.next_state_prediction
            self.next_state_prediction = self.next_state_prediction.copied_with(prediction=center_frame.worldstate)
            initial_frame.worldstate = self.world.state.state_replaced(self.next_state_prediction)

    def load(self, max_n, checkpoint_dict, preload_n, session, logf):
        # Control Force Estimator (CFE)
        if 'CFE' in checkpoint_dict:
            ik_checkpoint = os.path.expanduser(checkpoint_dict['CFE'])
            logf("Loading CFE from %s..." % ik_checkpoint)
            session.restore(ik_checkpoint, scope='CFE')
        # Observation Predictors (OP)
        n = 2
        while n <= max_n:
            if n == max_n and not preload_n: return
            checkpoint_path = None
            i = n
            while not checkpoint_path:
                if "OP%d"%i in checkpoint_dict:
                    checkpoint_path = os.path.expanduser(checkpoint_dict["OP%d"%i])
                else:
                    i //= 2
            if i == n:
                logf("Loading OP%d from %s..." % (n, checkpoint_path))
                session.restore(checkpoint_path, scope="OP%d" % n)
            else:
                logf("Loading OP%d from OP%d checkpoint from %s..." % (n, i, checkpoint_path))
                session.restore.restore_new_scope(checkpoint_path, "OP%d" % i, "OP%d" % n)
            n *= 2


    def load_all_from(self, max_n, ik_checkpoint, sm_checkpoint, sm_n, session, logf):
        # IK
        logf("Loading IK checkpoint from %s..." % ik_checkpoint)
        session.restore(ik_checkpoint, scope="ik")
        # SM
        n = 2
        while n <= max_n:
            source_n = sm_n(n) if callable(sm_n) else sm_n
            logf("Loading SM%d weights from SM%d checkpoint from %s..." % (n, source_n, sm_checkpoint))
            session.restore_new_scope(sm_checkpoint, "sm%d" % source_n, "sm%d" % n)
            n *= 2


@struct.definition()
class NextStatePrediction(State):

    def __init__(self, prediction, tags=('next_state_prediction',), name='next', **kwargs):
        State.__init__(self, **struct.kwargs(locals()))

    @struct.variable()
    def prediction(self, prediction):
        assert prediction is None or isinstance(prediction, StateCollection)
        return prediction

    def __repr__(self):
        return self.__class__.__name__
