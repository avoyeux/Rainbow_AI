"""
To run the different models on the data.
"""

import os
import gc

# Python codes imports
from AI_data_gen import DataGen
from AI_plotting import Plotter
from common_alf import decorators
from AI_models import UNetModel, UNetNDeepResidual

import tensorflow as tf

nb_cpus = 64
tf.config.threading.set_intra_op_parallelism_threads(nb_cpus)
tf.config.threading.set_inter_op_parallelism_threads(nb_cpus)

class ModelRunner:
    """
    Creates the data and runs the code for all the models. It is the main class.
    """

    def __init__(self, ModelClassList, epochs_list=[200], kernel_sizes=[(3, 3)], image_size=512, mask_val=0):
        # Arguments
        self.ModelClassList = ModelClassList
        self.epochs_list = epochs_list
        self.kernel_sizes = kernel_sizes
        self.image_size = image_size
        self.mask_val = mask_val

        # Functions
        self.Data()
        self.Running()

    def Data(self):
        """
        Generating the training and test data sets.
        """

        data = DataGen(mask_val=self.mask_val)
        self.train_images, self.train_masks, self.test_images, self.test_masks = data.Give_data()
    
    def Running(self):
        """
        Running the code given the different arguments. Here you can choose which model to compute
        """

        kwargs = {'train_images': self.train_images, 'train_masks': self.train_masks, 
                'test_images': self.test_images, 'test_masks': self.test_masks, 'image_size': 512,
                'mask_val': 0}

        for kernel_size in self.kernel_sizes:
            for epochs in self.epochs_list:
                kwargs['epochs'] = epochs
                kwargs['kernel_size'] = kernel_size

                # Running the different models
                for Model in self.ModelClassList:
                    Controller(model_class=Model, **kwargs)

@decorators.running_time
class Controller:
    """
    To compile, save and visualise the tests of a given model.
    """

    def __init__(self, model_class, train_images, train_masks, test_images, test_masks,
                 optimizer='adam', loss='binary_crossentropy', metrics=['acc'], 
                 epochs=200, kernel_size=(3, 3), image_size=512, result_cap=0.1, mask_val=0):

        # The model class
        self.model_class = model_class

        # Data inputs, outputs and tests
        self.train_images = train_images
        self.train_masks = train_masks
        self.test_images = test_images
        self.test_masks = test_masks

        # Model specific arguments
        self.model_optimizer = optimizer
        self.model_loss = loss
        self.model_metrics = metrics
        self.epochs = epochs
        self.kernel_size = kernel_size  
        self.image_size = image_size
        self.mask_val = mask_val

        # Visualisation argument
        self.result_cap = result_cap

        # Attributes
        self.path = os.path.join(os.getcwd(), self.model_class.__name__, 'first_layer32',
                                 f'kernel{self.kernel_size[0]}_{self.kernel_size[1]}', 
                                 f'Epochs{self.epochs}', f'maksval{self.mask_val}')
        os.makedirs(self.path, exist_ok=True)

        # Functions
        self.Execute()
        self.Plot()

    def Execute(self):
        """
        It compiles the given model on the training set and saves it.
        """

        model_instance = self.model_class(image_size=self.image_size, kernel_size=self.kernel_size)
        self.model = model_instance.Model()

        self.model.compile(optimizer=self.model_optimizer, loss=self.model_loss, metrics=self.model_metrics)
        self.model.fit(self.train_images, self.train_masks, epochs=self.epochs)
        self.model.save_weights(os.path.join(self.path, 'model.h5')) 

    def Plot(self):
        """
        Uses the Plotter class to plot the testing set and the corresponding model prediction.
        """

        result = self.model.predict(self.test_images)
        mask_result = result > self.result_cap

        # Plotting
        Plotter(path=self.path, test_images=self.test_images, test_masks=self.test_masks, predicted_masks=mask_result)

        # Deleting the model
        del self.model, self.model_class
        gc.collect()


if __name__ == '__main__':

    # Running the code for different models
    kernel_sizes = [(3, 3), (5, 5), (7, 7), (9, 9)]
    epochs_list = [1, 10, 50, 100, 200, 300]
    models_list = [UNetModel, UNetNDeepResidual]

    ModelRunner(ModelClassList=models_list, kernel_sizes=kernel_sizes, epochs_list=epochs_list)

