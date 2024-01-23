# coding=utf-8
from phi.tf.flow import *

from .pde.pde_base import collect_placeholders_channels
from .sequences import StaggeredSequence, SkipSequence, LinearSequence
from .hierarchy import PDEExecutor


class ControlTraining(LearningApp):

    def __init__(self, n, pde, datapath, val_range, train_range,
                 trace_to_channel=None,
                 obs_loss_frames=(-1,),
                 trainable_networks=('CFE', 'OP2'),
                 sequence_class=StaggeredSequence,
                 batch_size=16,
                 view_size=16,
                 learning_rate=1e-3,
                 learning_rate_half_life=1000,
                 dt=1.0,
                 new_graph=True):
        """

        :param n:
        :param pde:
        :param datapath:
        :param sequence_matching:
        :param train_cfe:
        """
        if new_graph:
            tf.reset_default_graph()
        LearningApp.__init__(self, 'Control Training', 'Train PDE control: OP / CFE', training_batch_size=batch_size,
                             validation_batch_size=batch_size, learning_rate=learning_rate, stride=50)
        self.initial_learning_rate = learning_rate
        self.learning_rate_half_life = learning_rate_half_life
        if n <= 1:
            sequence_matching = False
        diffphys = sequence_class is not None
        if sequence_class is None:
            assert 'CFE' not in trainable_networks, 'CRE training requires a sequence_class.'
            assert len(obs_loss_frames) > 0, 'No loss provided (no obs_loss_frames and no sequence_class).'
            sequence_class = SkipSequence
        self.n = n
        self.dt = dt
        self.data_path = datapath
        self.checkpoint_dict = None
        self.info('Sequence class: %s' % sequence_class)

        # --- Set up PDE sequence ---
        world = World(batch_size=batch_size)
        pde.create_pde(world, 'CFE' in trainable_networks, sequence_class != LinearSequence)  # TODO BATCH_SIZE=None
        world.state = pde.placeholder_state(world, 0)
        self.add_all_fields('GT', world.state, 0)
        target_state = pde.placeholder_state(world, n*dt)
        self.add_all_fields('GT', target_state, n)
        in_states = [world.state] + [None] * (n-1) + [target_state]
        for frame in obs_loss_frames:
            if in_states[frame] is None:
                in_states[frame] = pde.placeholder_state(world, frame*self.dt)
        # --- Execute sequence ---
        executor = self.executor = PDEExecutor(world, pde, target_state, trainable_networks, self.dt)
        sequence = self.sequence = sequence_class(n, executor)
        sequence.execute()
        all_states = self.all_states = [frame.worldstate for frame in sequence if frame is not None]
        # --- Loss ---
        loss = 0
        reg = None
        if diffphys:
            target_loss = pde.target_matching_loss(target_state, sequence[-1].worldstate)
            self.info('Target loss: %s' % target_loss)
            if target_loss is not None:
                loss += target_loss
            reg = pde.total_force_loss([state for state in all_states if state is not None])
            self.info('Force loss: %s' % reg)
        for frame in obs_loss_frames:
            supervised_loss = pde.target_matching_loss(in_states[frame], sequence[frame].worldstate)
            if supervised_loss is not None:
                self.info('Supervised loss at frame %d: %s' % (frame, supervised_loss))
                self.add_scalar('GT_obs_%d' % frame, supervised_loss)
                self.add_all_fields('GT', in_states[frame], frame)
                loss += supervised_loss
        self.info('Setting up loss')
        if loss is not 0:
            self.add_objective(loss, 'Loss', reg=reg)
        for name, scalar in pde.scalars.items():
            self.add_scalar(name, scalar)
        # --- Training data ---
        self.info('Preparing data')
        placeholders, channels = collect_placeholders_channels(in_states, trace_to_channel=trace_to_channel)
        data_load_dict = {p: c for p, c in zip(placeholders, channels)}
        self.set_data(data_load_dict,
                      val=None if val_range is None else Dataset.load(datapath, val_range),
                      train=None if train_range is None else Dataset.load(datapath, train_range))
        # --- Show all states in GUI ---
        for i, (placeholder, channel) in enumerate(zip(placeholders, channels)):
            def fetch(i=i): return self.viewed_batch[i]
            self.add_field('%s %d' % (channel, i), fetch)
        for i, worldstate in enumerate(all_states):
            self.add_all_fields('Sim', worldstate, i)
        for name, field in pde.fields.items():
            self.add_field(name, field)

    def add_all_fields(self, prefix, worldstate, index):
        with struct.unsafe():
            fields = struct.flatten(struct.map(lambda x: x, worldstate, trace=True))
        for field in fields:
            name = '%s[%02d] %s' % (prefix, index, field.path())
            if field.value is not None:
                self.add_field(name, field.value)
            # else:
            #     self.info('Field %s has value None' % name)

    def load_checkpoints(self, checkpoint_dict):
        if not self.prepared:
            self.prepare()
        self.checkpoint_dict = checkpoint_dict
        self.executor.load(self.n, checkpoint_dict, preload_n=True, session=self.session, logf=self.info)

    def action_save_model(self):
        self.save_model()

    def step(self):
        if self.learning_rate_half_life is not None:
            self.float_learning_rate = self.initial_learning_rate * 0.5 ** (self.steps / float(self.learning_rate_half_life))
        LearningApp.step(self)

    def infer_all_frames(self, data_range):
        dataset = Dataset.load(self.data_path, data_range)
        reader = BatchReader(dataset, self._channel_struct)
        batch = reader[0:len(reader)]
        feed_dict = self._feed_dict(batch, True)
        inferred = self.session.run(self.all_states, feed_dict=feed_dict)
        return inferred

    def infer_scalars(self, data_range):
        dataset = Dataset.load(self.data_path, data_range)
        reader = BatchReader(dataset, self._channel_struct)
        batch = reader[0:len(reader)]
        feed_dict = self._feed_dict(batch, True)
        scalar_values = self.session.run(self.scalars, feed_dict, summary_key='val', merged_summary=self.merged_scalars, time=self.steps)
        scalar_values = {name: value for name, value in zip(self.scalar_names, scalar_values)}
        return scalar_values
