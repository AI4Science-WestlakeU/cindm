

class PartitioningExecutor(object):

    def create_frame(self, index, step_count):
        return SeqFrame(index, type=TYPE_KEYFRAME if index == 0 or index == step_count else TYPE_UNKNOWN)

    def execute_step(self, initial_frame, target_frame, sequence):
        # type: (SeqFrame, SeqFrame, int) -> None
        print("Execute -> %d" % (initial_frame.index + 1))
        assert initial_frame.type >= TYPE_REAL
        target_frame.type = max(TYPE_REAL, target_frame.type)

    def partition(self, n, initial_frame, target_frame, center_frame):
        # type: (int, SeqFrame, SeqFrame, SeqFrame) -> None
        print("Partition length %d sequence (from %d to %d) at frame %d" % (n, initial_frame.index, target_frame.index, center_frame.index))
        assert initial_frame.type != TYPE_UNKNOWN and target_frame.type != TYPE_UNKNOWN
        center_frame.type = TYPE_PLANNED


TYPE_UNKNOWN = 0
TYPE_PLANNED = 1
TYPE_REAL = 2
TYPE_KEYFRAME = 3


class SeqFrame(object):

    def __init__(self, index, type=TYPE_UNKNOWN):
        self.index = index
        self.type = type

    def next(self):
        return self.index + 1

    def __repr__(self):
        return "Frame#%d" % self.index


class PartitionedSequence(object):

    def __init__(self, step_count, executor):
        # type: (int, PartitioningExecutor) -> None
        self.step_count = step_count
        self.executor = executor
        self._frames = [executor.create_frame(i, step_count) for i in range(step_count + 1)]

    def execute(self):
        self.partition_execute(self.step_count, 0)

    def partition_execute(self, n, start_frame_index, **kwargs):
        if n == 1:
            self.leaf_execute(self._frames[start_frame_index], self._frames[start_frame_index+1], **kwargs)
        else:
            self.branch_execute(n, start_frame_index, **kwargs)

    def leaf_execute(self, start_frame, end_frame, **kwargs):
        self.executor.execute_step(start_frame, end_frame, self)

    def branch_execute(self, n, start_frame_index, **kwargs):
        raise NotImplementedError()

    def partition(self, n, start_frame_index):
        self.executor.partition(n, self._frames[start_frame_index], self._frames[start_frame_index + n],
                                self._frames[start_frame_index + n // 2])

    def __getitem__(self, item):
        return self._frames[item]

    def __len__(self):
        return len(self._frames)

    def __iter__(self):
        return self._frames.__iter__()


class StaggeredSequence(PartitionedSequence):

    def __init__(self, step_count, executor):
        PartitionedSequence.__init__(self, step_count, executor)

    def branch_execute(self, n, start_frame_index, **kwargs):
        self.partition(n, start_frame_index)
        self.partition_execute(n//2, start_frame_index)
        self.partition_execute(n//2, start_frame_index+n//2)


class RefinedSequence(PartitionedSequence):

    def __init__(self, step_count, executor):
        PartitionedSequence.__init__(self, step_count, executor)

    def branch_execute(self, n, start_frame_index, update_target=False, **kwargs):
        self.partition(n, start_frame_index)
        self.partition_execute(n // 2, start_frame_index, update_target=True)
        if update_target:
            self.partition(n, start_frame_index + n)
            self.partition(n, start_frame_index + n // 2)
        self.partition_execute(n // 2, start_frame_index + n // 2, update_target=update_target)


class SkipSequence(PartitionedSequence):

    def __init__(self, sequence_length, executor):
        PartitionedSequence.__init__(self, sequence_length, executor)
        for i in range(1, sequence_length):
            if i != sequence_length//2:
                self._frames[i] = None

    def branch_execute(self, n, start_frame_index, **kwargs):
        self.partition(n, start_frame_index)


class LinearSequence(PartitionedSequence):

    def __init__(self, step_count, executor):
        PartitionedSequence.__init__(self, step_count, executor)

    def execute(self):
        for frame1, frame2 in zip(self._frames[:-1], self._frames[1:]):
            self.leaf_execute(frame1, frame2)

    def branch_execute(self, n, start_frame_index, **kwargs):
        raise AssertionError()
