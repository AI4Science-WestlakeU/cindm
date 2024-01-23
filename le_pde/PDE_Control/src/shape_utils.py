import os
import random

import numpy as np


def load_shapes(shape_dir):
    files = [os.path.join(shape_dir, file) for file in os.listdir(shape_dir) if file.endswith('.npz')]
    npzs = [np.load(file) for file in files]
    arrays = [npz[npz.files[0]][..., 0] for npz in npzs]
    return arrays


def distribute_random_shape(resolution, batch_size, shape_library, margin=1):
    array = np.zeros((batch_size,) + tuple(resolution) + (1,), np.float32)
    for batch in range(batch_size):
        shape = random.choice(shape_library)
        y = random.randint(margin, resolution[0] - margin - shape.shape[0] - 2)
        x = random.randint(margin, resolution[1] - margin - shape.shape[1] - 2)
        array[batch, y:(y + shape.shape[0]), x:(x + shape.shape[1]), 0] = shape
    assert array.dtype == np.float32
    return array

