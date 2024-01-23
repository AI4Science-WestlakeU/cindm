from phi.control.control import blur
from phi.tf.model import *
from phi.control.iksm import sm_resnet


class SupervisedSquareSM(TFModel):
    
    def __init__(self):
        TFModel.__init__(self, 'Supervised SquareSM', model_scope_name='sm2', data_validation_fraction=0.1)
        
        sim = self.sim = TFFluidSimulation([128] * 2, DomainBoundary([(False, True), (False, False)]), force_use_masks=True)
        initial_density = sim.placeholder(1, 'initial_density')
        center_density = sim.placeholder(1, 'center_density')
        target_density = sim.placeholder(1, 'target_density')
        
        with self.model_scope():
            prediction = sm_resnet(initial_density, target_density, training=self.training)
        
        loss = l2_loss(blur(center_density - prediction, self.editable_float("Blur_Radius", 2.0), 3))
        self.minimizer('Supervised_Loss', loss)

        self.database.add("initial_density", BatchSelect(0, "Density"))
        self.database.add("center_density", BatchSelect(8, "Density"))
        self.database.add("target_density", BatchSelect(16, "Density"))
        self.database.put_scenes(scenes('~/data/control/bounded-squares'), range(17), logf=self.info)
        self.database.put_scenes(scenes('~/data/control/rising-squares', max_count=1000), range(17), logf=self.info)
        self.finalize_setup([initial_density, center_density, target_density])

        self.add_field("Prediction", prediction)
        self.add_field("Ground Truth", "center_density")


SupervisedSquareSM().show()