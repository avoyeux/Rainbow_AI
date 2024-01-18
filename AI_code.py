"""
Main structure for the AI testing of the non RNN CNN models.
It imports the different created models, the data generator and the plotting function to save the results, the model and the weights
corresponding to each test.
"""

# Imports
import os
import gc

import numpy as np
# import tensorflow as tf

from AI_data_gen import DataGen
from AI_plotting import Plotter
from typeguard import typechecked
from common_alf import decorators
from AI_models import UNetModel, UNetNDeepResidual

# Parallelism tests to try and not use all the cpu cores. It failed though, but keeping it for later tries
# nb_cpus = 64
# tf.config.threading.set_intra_op_parallelism_threads(nb_cpus)
# tf.config.threading.set_inter_op_parallelism_threads(nb_cpus)


class ModelRunner:
    """
    The class that encompasses everything.
    Uses a list of model classes as inputs to run them given a list of arguments. Also runs the data generator class. 
    """

    @typechecked  # checking that the arguments types are right at runtime
    def __init__(self, ModelClassList: list, epochs_list: list = [200], kernel_sizes: list = [(3, 3)], image_size: int = 512):
        
        # Arguments
        self.ModelClassList = ModelClassList # list of the U-Net CNN model classes
        self.epochs_list = epochs_list   # list of the number of epochs to be tested
        self.kernel_sizes = kernel_sizes  # list of tuples of the kernel size to be tested
        self.image_size = image_size  # pixel length of the post processed square images

        # Created attributes
        self.train_inputs = None  # array of the training input images 
        self.train_outputs = None  # same of the outputs
        self.test_inputs = None  # array of the test input images 
        self.test_outputs = None  # same for the outputs
        
        # Functions
        self.Data()
        self.Running()

    def Data(self):
        """
        Generating the training and test data sets.
        """

        data = DataGen()
        self.train_images, self.train_masks, self.test_images, self.test_masks = data.Give_data()
    
    def Running(self):
        """
        Running the code given the different arguments.
        """

        kwargs = {'train_images': self.train_images, 'train_masks': self.train_masks, 
                'test_images': self.test_images, 'test_masks': self.test_masks, 'image_size': 512}

        for kernel_size in self.kernel_sizes:
            for epochs in self.epochs_list:
                kwargs['epochs'] = epochs
                kwargs['kernel_size'] = kernel_size

                # Running the different models
                for Model in self.ModelClassList:
                    Controller(model_class=Model, **kwargs)


class Controller:
    """
    To compile, save and visualise the tests of a given model.
    """

    @decorators.running_time
    def __init__(self, model_class: type, train_images: np.ndarray, train_masks: np.ndarray, test_images: np.ndarray, test_masks: np.ndarray,
                 optimizer: str = 'adam', loss: str = 'binary_crossentropy', metrics: list = ['acc'], epochs: int = 200, 
                 kernel_size: tuple = (3, 3), image_size: int = 512, result_cap: float = 0.1):

        # The model class
        self.model_class = model_class  # the model class object

        # Data inputs, outputs and tests
        self.train_images = train_images  # array of the training input images 
        self.train_masks = train_masks  # same of the outputs
        self.test_images = test_images  # array of the test input images 
        self.test_masks = test_masks   # same for the outputs

        # Model specific arguments
        self.model_optimizer = optimizer  # the optimiser used for the model
        self.model_loss = loss  # the loss function used for the model
        self.model_metrics = metrics  # the metrics type used for the model
        self.epochs = epochs  # the number of epochs
        self.kernel_size = kernel_size  # tuple representing the kernel size to be tested
        self.image_size = image_size  # pixel length of the post processed square images

        # Visualisation argument
        self.result_cap = result_cap  # float representing the value at which something is flagged as a mask (i.e. 1) in the output plot.

        # Created attributes
        self.path = None  # string path of where the plots are saved
        self.model = None  # instance of the model class

        # Functions
        self.Paths()
        self.Execute()
        self.Plot()

    def Paths(self):
        """
        To created the needed paths. Here it is only the output path.
        """

        self.path = os.path.join(os.getcwd(), self.model_class.__name__, f'kernel{self.kernel_size[0]}_{self.kernel_size[1]}', 
                                 f'Epochs{self.epochs}')
        os.makedirs(self.path, exist_ok=True)

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

