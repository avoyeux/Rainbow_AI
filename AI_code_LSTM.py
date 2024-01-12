"""
Main structure of the AI testing.
It imports the different created models, the data generator and the plotting function to save the results, the model and the weights
corresponding to each test.
"""

import os
import gc
import numpy as np

# Python codes imports
from AI_data_gen import DataGen_ordered
from AI_plotting import Plotter_LSTM
from common_alf import decorators
from AI_models import UNetConvLSTM2D_long, UNetConvLSTM2D_short

import tensorflow as tf

# Parallelism tests to try and not use all the cpu cores. It failed though, but keeping it for later tries
nb_cpus = 92
tf.config.threading.set_intra_op_parallelism_threads(nb_cpus)
tf.config.threading.set_inter_op_parallelism_threads(nb_cpus)

class ModelRunner:
    """
    The class that encompasses everything.
    Uses a list of model classes as inputs to run them given a list of arguments. Also runs the data generator class. 
    """

    def __init__(self, ModelClassList: list, epochs_list: list = [200], kernel_sizes: list = [(3, 3)], image_size: int = 512, sequence_len: int = 8,
                  mask_val: int = 0):
        # Arguments
        self.ModelClassList = ModelClassList
        self.epochs_list = epochs_list
        self.kernel_sizes = kernel_sizes
        self.image_size = image_size
        self.sequence_len = sequence_len
        self.mask_val = mask_val

        # Functions
        self.Data()
        self.Running()

    def Data(self):
        """
        Generating the training and test data sets.
        """
        
        kwargs = {'image_size': self.image_size, 'mask_val': self.mask_val, 'sequence_len': self.sequence_len}

        data = DataGen_ordered(**kwargs)
        self.train_inputs, self.train_outputs, self.test_inputs, self.test_outputs = data.Give_data()
    
    def Running(self):
        """
        Running the code given the different arguments.
        """

        kwargs = {'train_images': self.train_inputs, 'train_masks': self.train_outputs, 
                'test_images': self.test_inputs, 'test_masks': self.test_outputs, 'image_size': self.image_size,
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
    To compile, save and visualise the tests for a given model as input.
    """

    def __init__(self, model_class, train_images, train_masks, test_images, test_masks, sequence_len=8, batch_size=5,
                 optimizer='adam', loss='binary_crossentropy', metrics=['acc'], 
                 epochs=200, kernel_size=(3, 3), image_size=512, result_cap=0.1, mask_val=0):

        # The model class
        self.model_class = model_class

        # Data inputs, outputs and tests
        self.train_images = train_images
        self.train_masks = train_masks
        self.test_images = test_images
        self.test_masks = test_masks
        self.sequence_len = sequence_len
        self.batch_size = batch_size

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
        self.path = os.path.join(os.getcwd(), self.model_class.__name__, f'seq_len{self.sequence_len}', 'firstlayer64',
                                 f'kernel{self.kernel_size[0]}_{self.kernel_size[1]}', 
                                 f'Epochs{self.epochs}', 'no_init_mask')
        os.makedirs(self.path, exist_ok=True)

        # Functions
        self.Execute()
        self.Plot()

    def Execute(self):
        """
        It compiles the given model on the training set and saves it.
        """

        model_instance = self.model_class(sequence_len=self.sequence_len, image_size=self.image_size, kernel_size=self.kernel_size)
        self.model = model_instance.Model()

        self.model.compile(optimizer=self.model_optimizer, loss=self.model_loss, metrics=self.model_metrics)
        self.model.fit(self.train_images, self.train_masks, batch_size=self.batch_size, epochs=self.epochs)
        json_config = self.model.to_json()
        with open(os.path.join(os.getcwd(), self.model_class.__name__, 'model_config.json'), 'w') as json_file:
            json_file.write(json_config)
        self.model.save_weights(os.path.join(self.path, 'model.h5')) 

    def Plot(self):
        """
        Uses the Plotter class to plot the testing set and the corresponding model prediction.
        """

        test_image = tf.expand_dims(self.test_images[0], axis=0)
        result0 = self.model.predict(test_image)
        mask_result0 = result0 > self.result_cap

        test_image = tf.expand_dims(self.test_images[1], axis=0)
        result1 = self.model.predict(test_image)
        mask_result1 = result1 > self.result_cap

        test_image = tf.expand_dims(self.test_images[2], axis=0)
        result2 = self.model.predict(test_image)
        mask_result2 = result2 > self.result_cap

        mask_result = []
        mask_result.extend([mask_result0, mask_result1, mask_result2])

        self.test_masks = tf.expand_dims(self.test_masks, axis=3)
        self.test_masks = np.array(self.test_masks)

        # Plotting
        Plotter_LSTM(path=self.path, test_images=self.test_images, test_masks=self.test_masks, predicted_masks=mask_result)

        # Deleting the model
        del self.model, self.model_class
        gc.collect()


if __name__ == '__main__':

    # Running the code for different models
    kernel_sizes = [(3, 3), (5, 5), (7, 7), (9, 9)]
    epochs_list = [1, 5, 10, 25, 50, 100, 200]
    models_list = [UNetConvLSTM2D_short]

    ModelRunner(ModelClassList=models_list, kernel_sizes=kernel_sizes, epochs_list=epochs_list)

