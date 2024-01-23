# coding=utf-8
from __future__ import print_function
import sys
from phi.tf.model import *
from phi.control.control import *
from phi.control.iksm import *


class SmokeSM(TFModel):
    def __init__(self):
        TFModel.__init__(self, 'SmokeSM %d'%n, 'Slow motion density generation CNN training',
                         learning_rate=1e-3, data_validation_fraction=0.1,
                         training_batch_size=4, validation_batch_size=100,
                         model_scope_name='sm%d'%n)
        self.info('Mode: %s' % ('Graph' if use_graph else 'Inference'))
        self.info('Preload: %s' % ('Yes' if preload else 'No'))
        self.info('Load frames: %s' % ('All' if load_all_frames else 'Start/End/Center'))
        self.info('Divide Strategy: %s' % divide_strategy)
        self.info('Force: %s' % force_inference)
        self.info('Autorun: %s' % (('Supervised' if supervised else 'Unsupervised') if autorun else 'No'))

        sim = self.sim = TFFluidSimulation([64 if half_res else 128] * 2, 'open')

        initial = sim.placeholder(name='initial')
        center = sim.placeholder(name='center')
        target = sim.placeholder(name='target')
        with self.model_scope():
            prediction = sm_resnet(initial, target)
        loss = l2_loss(prediction - center)
        self.minimizer('Supervised_Loss', loss)

        self.database.add('initial', BatchSelect(lambda len: range(len - n), 'Density'))
        self.database.add('center', BatchSelect(lambda len: range(n//2, len - n//2), 'Density'))
        self.database.add('target', BatchSelect(lambda len: range(n, len), 'Density'))
        if n <= 8:
            self.database.put_scenes(scenes('~/data/SmokeIK', 'random_128x128'), per_scene_indices=range(4, 4+n+1), logf=self.info)
        else:
            self.database.put_scenes(scenes('~/data/SmokeIK', 'random3p_128x128'), per_scene_indices=range(4, 4+n+1), logf=self.info)  # Something in this dataset causes NaN for small n
        self.finalize_setup([initial, center, target])

        self.add_field('Initial', 'initial')
        self.add_field('Center', 'center')
        self.add_field('Target', 'target')
        self.add_field('Frame', 'frame')
        self.add_field('Prediction', prediction)

if 'help' in sys.argv or '-help' in sys.argv or '--help' in sys.argv:
    print('First argument: n (integer)')
    print('Keywords: inference, preload, all_frames, adaptive, autorun, unsupervised, exact_force, forcenet')
    exit(0)
if len(sys.argv) >= 2:
    n = int(sys.argv[1])
else:
    n = 2
use_graph = 'inference' not in sys.argv
preload = 'preload' in sys.argv or not use_graph
load_all_frames = 'all_frames' in sys.argv
divide_strategy = 'adaptive' if 'adaptive' in sys.argv else 'binary'
autorun = 'autorun' in sys.argv
supervised = 'unsupervised' not in sys.argv
force_inference = 'exact' if 'exact_force' in sys.argv else 'advection'
half_res = 'half_res' in sys.argv
if 'forcenet' in sys.argv: force_inference = 'forcenet'

app = SmokeSM()
if autorun: app.play()
app.show(display=('Prediction', 'Center'), production=__name__ != '__main__')