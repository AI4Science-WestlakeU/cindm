# Legacy code

The original code including PhiFlow 0.2.




## Naming differences between paper and code

| Name in paper                 | Name in code            |
|-------------------------------|-------------------------|
| Observation predictor (OP)    | Slow motion (SM)        |
| Control force estimator (CFE) | Inverse kinematics (IK) |
| Staggered execution           | Binary tree             |
| Prediction refinement         | Interleaved tree        |
| Reconstructed trajectory      | Real sequence           |



## Code by experiment

The followings apps, located in the `apps` folder, were used for training and running the neural networks.
Code for plotting and evaluation of intermediate results is located in `paper`.


### 1. Burger

Data generation: `burgergen.py`

Supervised CFE training: `burgercfe_supervised.py`

Diff-Phys. CFE training and evaluation: `burgercfe_diffphys.py`

Hierarchical pre-training: `burgersm.py`

Hierarchical training: `burgersm_refine.py`




### 2 Incompressible fluid

Train CFE:
`smokeik.py`

Supervised OP pre-training
`smokesm_supervised.py`

Pre-train OPs (supervised + diff-phys.):
`smokesm.py`

Train OPs with diff-phys.:
`smokesm_refine.py`

Evaluate results:
`smokesm_eval.py`



### 2.1 Natural flow

Short sequence data generation:
`smokegen_simple.py`

Long sequence data generation:
`smokegen_three_pass.py`


### 2.2 Shapes

Data generation (moving squares):
`smokegen_blob.py`

Data generation (random shapes):
`shapegen.py`


### 2.3 Multiple shapes

Data generation ("i"-sequence for paper):
`shapegen_specific.py`

Evaluate results:
`smokesm_multishape.py`


### 2.4 Classical optimization

Classical optimization:
`smokeoverfit.py`




### 3. Indirect control

Training data generation (squares moving in the inner region in any direction):
`squaregen_buckets.py`

Training data generation (squares moving into one of three buckets at the top):
`squaregen_buckets_rising.py`

Train single-step CFE:
`smokeik_indirect_training.py`

Train multi-step CFE:
`smokeik_indirect_refine.py`

Pretrain OPs to move square in a straight line:
`train_supervised_squaresm.py`

Train OPs:
`smokesm_indirect_refine.py`
