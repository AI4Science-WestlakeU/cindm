import numpy as np

from phi.flow import Scene

from control.control_training import ControlTraining
from control.pde.value import IncrementPDE


# Data generation
from control.sequences import StaggeredSequence

if len(Scene.list('~/phi/data/value')) == 0:
    scene = Scene.create('~/phi/data/value')
    for frame in range(32):
        scene.write_sim_frame([np.zeros([1])+frame], ['data'], frame)

app = ControlTraining(n=8,
                      pde=IncrementPDE(),
                      datapath='~/phi/data/value',
                      val_range=range(1),
                      train_range=None,
                      obs_loss_frames=[-1],
                      trainable_networks=[],
                      sequence_class=StaggeredSequence,
                      batch_size=1)

print("Training app was set up. The values of 'scalar.data' should be equal to the frame index. 'fieldeffect_xxx_.field.data' should be 1 for frame>=1.")
for fieldname in app.fieldnames:
    if 'Sim' in fieldname:
        value = app.get_field(fieldname)
        print("%s = %s" % (fieldname, value[..., 0]))
