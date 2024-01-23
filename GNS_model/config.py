from yacs.config import CfgNode

_C = CfgNode()
_C.TRAIN_DIR = None
_C.VAL_DIR = None
_C.N_HIS = 4
_C.ROLLOUT_STEPS = 240
_C.PRED_STEPS = 1
_C.DATASET_ABS = None
_C.MAX_VAL = 30
_C.KINEMATIC_PARTICLE_ID = 1
_C.NUM_PARTICLE_TYPES = 2

_C.SOLVER = CfgNode()
_C.SOLVER.BASE_LR = 1e-4
_C.SOLVER.MIN_LR =1e-6
_C.SOLVER.VAL_INTERVAL = 25000
_C.SOLVER.WEIGHT_DECAY = 0.0
_C.SOLVER.WARMUP_ITERS = -1
_C.SOLVER.MAX_ITERS = 5e6
_C.SOLVER.BATCH_SIZE = 1
_C.SOLVER.LR_DECAY_INTERVAL = 5e6


_C.NET = CfgNode()
_C.NET.RADIUS = 0.2 
_C.NET.NOISE = 0.00000067
_C.NET.PARTICLE_EMB_SIZE = 2
_C.NET.MAX_EDGE_PER_PARTICLE =6
_C.NET.SELF_EDGE = False
_C.NET.NODE_FEAT_DIM_IN =8
_C.NET.EDGE_FEAT_DIM_IN = 3
_C.NET.GNN_LAYER = 5
_C.NET.HIDDEN_SIZE = 64
_C.NET.OUT_SIZE = 2
_C.NET.recover_what = 'vel' # ['acc', vel', 'pos']
