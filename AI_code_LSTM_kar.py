"""
Main structure for the AI testing of the RNN (here LSTM) models for karine's masks.
It imports the different created models, the data generator and the plotting function to save the results, the model and the weights
corresponding to each test.
"""

# Imports
import os
import gc
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from typeguard import typechecked
from common_alf import decorators
from keras.models import load_model
from keras.backend import clear_session
# from AI_plotting import Plotter_LSTM
from keras.mixed_precision import set_global_policy
from keras.callbacks import EarlyStopping, ModelCheckpoint
from AI_models_kar import VaryingLSTMKernelSizes, VaryingLSTMKernelSizes_2Inputs
from AI_data_gen_kar import KarineDataGenOrdered, KarineDataGenOrdered_2Inputs, KarineDataGenOrdered_1Input
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error

# Parallelism tests to try and not use all the cpu cores. It failed though, but keeping it for later tries
# nb_cpus = 92
# tf.config.threading.set_intra_op_parallelism_threads(nb_cpus)
# tf.config.threading.set_inter_op_parallelism_threads(nb_cpus)


class ModelRunner:
    """
    The class that encompasses everything.
    Uses a list of model classes as inputs to run them given a list of arguments. Also runs the data generator class. 
    """

    @typechecked  # checking that the arguments types are right at runtime
    def __init__(self, ModelClassList: list, DataClass: type, kernel_size_LSTM: list, kernel_size_conv: list, epochs: int = 500,  
                 image_size: int = 512, sequence_len: int = 8, first_filters: int = 64, batch_size: int = 5):
        
        # Arguments
        self.ModelClassList = ModelClassList  # list of the U-Net LSTM model classes
        self.DataClass = DataClass  # the class used for the data gen
        self.epochs = epochs  # list of the number of epochs to be tested
        self.kernel_size_conv = kernel_size_conv
        self.kernel_size_LSTM = kernel_size_LSTM
        self.image_size = image_size  # pixel length of the post processed square images
        self.sequence_len = sequence_len  # number of images used for each sequences (i.e. the time dependence length)
        self.first_filters = first_filters  # initial nb of filters for the U-Net
        self.batch_size = batch_size

        # Created attributes
        self.train_inputs = None  # array of the training input images 
        self.train_outputs = None  # same of the outputs
        self.validation_inputs = None  # array of the validation input images
        self.validation_outputs = None  # same for the outputs
        self.test_inputs = None  # array of the test input images 
        self.test_outputs = None  # same for the outputs

        # Functions
        self.Data()
        self.Running()

    def Data(self):
        """
        Generating the training and test data sets.
        """
        
        kwargs = {'image_size': self.image_size, 'sequence_len': self.sequence_len}

        data = self.DataClass(**kwargs)
        self.train_inputs, self.train_outputs, self.validation_inputs, self.validation_outputs, \
            self.test_inputs, self.test_outputs = data.Give_data()

        self.DataClass = None
    
    def Running(self):
        """
        Running the code given the different arguments.
        """

        kwargs = {'train_images': self.train_inputs, 'train_masks': self.train_outputs, 
                  'validation_images': self.validation_inputs, 'validation_masks': self.validation_outputs,
                  'test_images': self.test_inputs, 'test_masks': self.test_outputs, 'image_size': self.image_size, 
                  'epochs': self.epochs, 'kernel_size_conv': self.kernel_size_conv, 'kernel_size_LSTM': self.kernel_size_LSTM,
                  'first_filters': self.first_filters, 'batch_size': self.batch_size}

        # Running the different models
        for Model in self.ModelClassList:
            Controller(ModelClass=Model, **kwargs)


class Controller:
    """
    To compile, save and visualise the tests for a given model as input.
    """

    @decorators.running_time
    def __init__(self, ModelClass: type, kernel_size_conv: list, kernel_size_LSTM: list, train_images: np.ndarray, train_masks: np.ndarray, 
                 validation_images: np.ndarray, validation_masks: np.ndarray, test_images: np.ndarray, test_masks: np.ndarray,
                 sequence_len: int = 8, batch_size: int = 5, optimizer: str = 'adam', loss: str = 'binary_crossentropy', metrics: list = ['acc'], 
                 epochs: int = 500, image_size: int = 512, result_cap: float = 0.5, first_filters: int = 64):

        # The model class
        self.ModelClass = ModelClass # the model class object

        # Data inputs, outputs and tests
        self.train_images = train_images  # array of the training input images 
        self.train_masks = train_masks  # same of the outputs
        self.validation_images = validation_images  # array of the validation images
        self.validation_masks = validation_masks  # same for the outputs
        self.test_images = test_images  # array of the test input images 
        self.test_masks = test_masks  # same for the outputs
        self.sequence_len = sequence_len  # number of images used for each sequences (i.e. the time dependence length)
        self.batch_size = batch_size  # the batch size used in the model training

        # Model specific arguments
        self.model_optimizer = optimizer  # the optimiser used for the model
        self.model_loss = loss  # the loss function used for the model
        self.model_metrics = metrics  # the metrics type used for the model
        self.epochs = epochs  # the number of epochs
        self.kernel_size_conv = kernel_size_conv  # tuple representing the kernel size to be tested for the Conv2D
        self.kernel_size_LSTM = kernel_size_LSTM  # same for ConvLSTM2D
        self.image_size = image_size  # pixel length of the post processed square images
        self.first_filters = first_filters  # initial nb of filters in the U-Net

        # Visualisation argument
        self.result_cap = result_cap  # float representing the value at which something is flagged as a mask (i.e. 1) in the output plot.

        # Created attributes
        self.path = None  # string path of where the plots are saved
        self.model = None  # instance of the model class

        # Functions
        self.Paths()
        self.Execute()
        self.Statistics()

    def Paths(self):
        """
        To created the needed paths. Here it is only the output path.
        """

        self.path = os.path.join(os.getcwd(), self.ModelClass.__name__, f'filters{self.first_filters}', f'seq_len{self.sequence_len}', 
                                 f'lstm{self.kernel_size_LSTM[0][0]}_{self.kernel_size_LSTM[1][0]}_{self.kernel_size_LSTM[2][0]}'
                                 f'conv{self.kernel_size_conv[0][0]}_{self.kernel_size_conv[1][0]}_{self.kernel_size_conv[2][0]}',
                                 f'batch{self.batch_size}')
        os.makedirs(self.path, exist_ok=True)

    def Execute(self):
        """
        It compiles the given model on the training set and saves it.
        """

        model_instance = self.ModelClass(sequence_len=self.sequence_len, image_size=self.image_size, 
                                         kernel_size_LSTM=self.kernel_size_LSTM, kernel_size_conv=self.kernel_size_conv, first_filters=self.first_filters)

        # For early stopping and choosing the best model
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        model_checkpoint = ModelCheckpoint(os.path.join(self.path, 'model.keras'), monitor='val_loss', mode='min', save_best_only=True, verbose=1)

        # Mixed precision policy 
        set_global_policy('mixed_float16')

        # Compiling and setting up the model on multiple GPUs
        strategy = MultiGPUs.Mirroring()
        with strategy.scope():
            self.model = model_instance.Model()
            self.model.compile(optimizer=self.model_optimizer, loss=self.model_loss, metrics=self.model_metrics)

        # Running the model
        self.start_time = time.time()
        history = self.model.fit(self.train_images, self.train_masks, batch_size=self.batch_size, epochs=self.epochs, 
                       validation_data=(self.validation_images, self.validation_masks),
                       callbacks=[early_stopping, model_checkpoint])
        self.best_epochs = np.argmin(history.history['val_loss']) + 1
        self.end_time = time.time()

        # Freeing up the GPU
        del self.model
        clear_session()
        gc.collect()
        print('Freeing up the VRAM')

    def VRAM_evaluate(self, images, masks):
        """
        To save some VRAM by uploading the model and by deleting it right after.
        Had to do this as the RAM fills up quite quickly when using a high number of filters.
        Probably doing something wrong somewhere, but here is the quick patch up.
        """

        best_model = load_model(os.path.join(self.path, 'model.keras'), safe_mode=False)
        loss, accuracy = best_model.evaluate(images, masks, batch_size=self.batch_size)
        del best_model  # probably don't need this line but just to be sure
        clear_session()
        gc.collect()
        return loss, accuracy
    
    def VRAM_prediction(self, images):
        """
        To save some VRAM by uploading the model and by deleting it right after.
        Had to do this as the RAM fills up quite quickly when using a high number of filters.
        Probably doing something wrong somewhere, but here is the quick patch up.
        """

        best_model = load_model(os.path.join(self.path, 'model.keras'), safe_mode=False)
        y_pred = best_model.predict(images, batch_size=self.batch_size)
        del best_model  # probably don't need this line but just to be sure
        clear_session()
        gc.collect()
        return y_pred

    def Statistics(self):
        """
        To save the stats for each model in a csv file.
        """

        training_time = self.end_time - self.start_time
        print(f'Best model uploaded')
        train_loss, train_accuracy = self.VRAM_evaluate(self.train_images, self.train_masks)  
        val_loss, val_accuracy = self.VRAM_evaluate(self.validation_images, self.validation_masks)
        test_loss, test_accuracy = self.VRAM_evaluate(self.test_images, self.test_masks)

        print('starting predictions:')
        train_y_pred = self.VRAM_prediction(self.train_images)
        print('training predictions done.')
        val_y_pred = self.VRAM_prediction(self.validation_images)
        print('validation predictions done.')
        test_y_pred = self.VRAM_prediction(self.test_images)
        print('testing predictions done.')

        train_y_pred = (train_y_pred > 0.5).astype('uint8').flatten()
        val_y_pred = (val_y_pred > 0.5).astype('uint8').flatten()
        test_y_pred = (test_y_pred > 0.5).astype('uint8').flatten()

        train_precision, train_recall, train_f1, train_mse, train_mae, train_roc_auc, train_iou = self.Stats_functions(self.train_masks.flatten(), train_y_pred)
        val_precision, val_recall, val_f1, val_mse, val_mae, val_roc_auc, val_iou = self.Stats_functions(self.validation_masks.flatten(), val_y_pred)
        test_precision, test_recall, test_f1, test_mse, test_mae, test_roc_auc, test_iou = self.Stats_functions(self.test_masks.flatten(), test_y_pred)
        print('stats finished.')
        metrics = {'Info/path': [self.path],
                   'Training Time (s)': [training_time], 'Epochs': [self.best_epochs], 'Train loss': [train_loss], 'Train accuracy': [train_accuracy],
                   'Validation loss': [val_loss], 'Validation accuracy': [val_accuracy], 'Test loss': [test_loss], 'Test accuracy': [test_accuracy],
                   'Train precision': [train_precision], 'Validation precision': [val_precision], 'Test precision': [test_precision], 
                   'Train recall': [train_recall], 'Validation recall': [val_recall], 'Test recall': [test_recall], 
                   'Train f1': [train_f1], 'Validation f1': [val_f1], 'Test f1': [test_f1],  
                   'Train mse': [train_mse], 'Validation mse': [val_mse], 'Test mse': [test_mse],
                   'Train mae': [train_mae], 'Validation mae': [val_mae], 'Test mae': [test_mae], 
                   'Train roc auc': [train_roc_auc], 'Validation roc auc': [val_roc_auc], 'Test roc auc': [test_roc_auc],
                   'Train iou': [train_iou], 'Validation iou': [val_iou], 'Test iou': [test_iou]}
        
        df = pd.DataFrame(metrics, index=[self.ModelClass.__name__])
        df_transposed = df.T

        df_final = df_transposed.reset_index()
        df_final.columns = ['Metric', self.ModelClass.__name__]
        df_final.to_csv(os.path.join(self.path, 'statistics.csv'), index=False)
        print('csv file saved.')

    def Stats_functions(self, labels, predictions):
        """
        Has the collection of the stats functions
        """

        precision = precision_score(labels, predictions, average='binary')
        recall = recall_score(labels, predictions, average='binary')
        f1 = f1_score(labels, predictions, average='binary')
        mse = mean_squared_error(labels, predictions)
        mae = mean_absolute_error(labels, predictions)
        auc_roc = roc_auc_score(labels, predictions)

        intersection = np.logical_and(labels, predictions)
        union = np.logical_or(labels, predictions)
        iou_score = np.sum(intersection) / np.sum(union)
        return precision, recall, f1, mse, mae, auc_roc, iou_score
    

class MultiGPUs:
    """
    To be able to choose multiple GPUs if there are multiple in the given node.
    This was done to have the maximum VRAM possible.
    """

    @staticmethod
    def Available_GPUs():
        """
        Gives the list of the identifies for the GPUs in the node.
        """

        gpus = tf.config.experimental.list_physical_devices('GPU')
        return [f'/gpu:{i}' for i in range(len(gpus))]
    
    @staticmethod
    def Mirroring():
        """
        TensorFlow MirroredStrategy based on available GPUs.
        """

        gpu_names = MultiGPUs.Available_GPUs()
        if not gpu_names:  # TODO: still wasn't able to make it work...
            print(f'Available GPU names are {gpu_names}.')
            strategy = tf.distribute.MirroredStrategy()
            return strategy
        else:
            print(f'No GPUs found.')
            return tf.distribute.get_strategy()



if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Running the code for different models
    ModelRunner(ModelClassList=[VaryingLSTMKernelSizes_2Inputs], DataClass=KarineDataGenOrdered_2Inputs, kernel_size_LSTM=[(5, 5), (5, 5), (5, 5)], 
                kernel_size_conv=[(3, 3), (3, 3), (3, 3)], epochs=1000, first_filters=128, batch_size=1)

