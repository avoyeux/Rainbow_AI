"""
To understand the statistics.csv file to hopefully get the best models.
"""

# Imports 
import os
import re
import glob 

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


class Statistics:
    """
    To run some statistics on the statistic.csv file.
    """

    def __init__(self, saving_csv: bool = True, opening_csv: bool = True):
        
        # Functions
        self.Paths()
        self.Patterns()

        if saving_csv:
            self.CSV_to_pandas()
        if opening_csv:
            self.Opening_total_csv()
        

    def Paths(self):
        """
        To create a path dictionary.
        """

        self.paths = {'main': os.getcwd(), 
                      'stats': glob.glob(os.path.join(os.getcwd(), 'VaryingLSTMKernelSizes_2Inputs', '**', 'statistics.csv'), recursive=True), 
                      'save': os.path.join(os.getcwd(),'VaryingLSTMKernelSizes_2Inputs', 'STATISTICS')}
        os.makedirs(self.paths['save'], exist_ok=True)

    def CSV_to_pandas(self):
        """
        To open and sort out the data from the .csv files.
        """
        print(f"the stats paths are {self.paths['stats']}")
        data_frames = [pd.read_csv(csv_filepath) for csv_filepath in self.paths['stats']]
        data_frames = pd.concat(data_frames, ignore_index=True)
        data_frames.to_csv(os.path.join(self.paths['save'], 'total_stats.csv'), index=False)

    def Opening_total_csv(self):
        """
        To open the created .csv in the last function which has all the data concatenated.
        """

        df = pd.read_csv(os.path.join(self.paths['save'], 'total_stats.csv'))

        names = self.Name_changing(df['Info/path'])
        print(f'names are {names}')
        # Increase is better:
        train_acc = self.Normalisation(df['Train accuracy'])
        val_acc = self.Normalisation(df['Validation accuracy'])
        test_acc = self.Normalisation(df['Test accuracy'])
        train_prec = self.Normalisation(df['Train precision'])
        val_prec = self.Normalisation(df['Validation precision'])
        test_prec = self.Normalisation(df['Test precision'])
        train_rec = self.Normalisation(df['Train recall'])
        val_rec = self.Normalisation(df['Validation recall'])
        test_rec = self.Normalisation(df['Test recall'])
        train_f1 = self.Normalisation(df['Train f1'])
        val_f1 = self.Normalisation(df['Validation f1'])
        test_f1 = self.Normalisation(df['Test f1'])
        train_roc = self.Normalisation(df['Train roc auc'])
        val_roc = self.Normalisation(df['Validation roc auc'])
        test_roc = self.Normalisation(df['Test roc auc'])
        train_iou = self.Normalisation(df['Train iou'])
        val_iou = self.Normalisation(df['Validation iou'])
        test_iou = self.Normalisation(df['Test iou'])

        # Decrease is better:
        train_loss = self.Normalisation(df['Train loss'])
        val_loss = self.Normalisation(df['Validation loss'])
        test_loss = self.Normalisation(df['Test loss'])
        train_mse = self.Normalisation(df['Train mse'])
        val_mse = self.Normalisation(df['Validation mse'])
        test_mse = self.Normalisation(df['Test mse'])
        train_mae = self.Normalisation(df['Train mae'])
        val_mae = self.Normalisation(df['Validation mae'])
        test_mae = self.Normalisation(df['Test mae'])

        # Data visualisation
        train_increasing_values = np.array((train_acc, train_prec, train_rec, train_f1, train_roc, train_iou))
        val_increasing_values = np.array((val_acc, val_prec, val_rec, val_f1, val_roc, val_iou))
        test_increasing_values = np.array((test_acc, test_prec, test_rec, test_f1, test_roc, test_iou))
        self.Plotting_increase(names, train_increasing_values, 'Training')
        self.Plotting_increase(names, val_increasing_values, 'Validation')
        self.Plotting_increase(names, test_increasing_values, 'Testing')

        train_decreasing_values = np.array((train_loss, train_mse, train_mae))
        val_decreasing_values = np.array((val_loss, val_mse, val_mae))
        test_decreasing_values = np.array((test_loss, test_mse, test_mae))
        self.Plotting_decrease(names, train_decreasing_values, 'Training')
        self.Plotting_decrease(names, val_decreasing_values, 'Validation')
        self.Plotting_decrease(names, test_decreasing_values, 'Testing')
    
    def Name_changing(self, names):
        """
        Using the pathname in the csv to get a more compact version of the arguments used.
        """

        compact_names = []
        for name in names:
            name_groups = self.pattern.match(name)
            if name_groups:
                compact_names.append(f"F{name_groups.group('nb_of_filters')}lstm{name_groups.group('lstm')}conv{name_groups.group('convolution')}")
            else:
                raise ValueError(f'No pattern match found for {name}') 
        return np.array(compact_names)

    def Patterns(self):
        """
        To use the pathname saved in the csv to get a more compact argument values.
        """


        self.pattern = re.compile(r'''/home/avoyeux/AI_tests/
                                  (?P<model_name>.+?)/
                                  filters(?P<nb_of_filters>\d+)/
                                  seq_len(?P<sequence_length>\d+)/
                                  lstm(?P<lstm>\d+_\d+_\d+)
                                  conv(?P<convolution>\d+_\d+_\d+)/
                                  batch(?P<batch_nb>\d+)''', re.VERBOSE)

    def Plotting_increase(self, names, values, data_type: str):
        """
        The plot for the higher=better metrics.
        """
        metric_names = ['Accuracy', 'Precision', 'Sensitivity', 'F1', 'ROC AUC', 'Intersection over union']

        # Ordering the values with the best at the end
        sum_array = np.sum(values, axis=0)
        sorted_indexes = np.argsort(sum_array)
        sorted_values = values[:, sorted_indexes]
        sorted_names = names[sorted_indexes]

        plt.figure()
        plt.title(f'The higher the better: {data_type}')

        for i, metric in enumerate(sorted_values):
            plt.plot(sorted_names, metric, label=metric_names[i])
        plt.legend()
        plt.savefig(f'Increasing_{data_type}.png', dpi=300)
        plt.close()

    def Plotting_decrease(self, names, values, data_type: str):
        """
        The plot for the lower=better metrics.
        """

        metric_names = ['Loss', 'Mean squared error', 'Mean absolute error']

        # Ordering the values with the best at the end
        sum_array = np.sum(values, axis=0)
        sorted_indexes = np.argsort(sum_array)[::-1]
        sorted_values = values[:, sorted_indexes]
        sorted_names = names[sorted_indexes]

        plt.figure()
        plt.title(f'The lower the better: {data_type}')

        for i, metric in enumerate(sorted_values):
            plt.plot(sorted_names, metric, label=metric_names[i])
        plt.legend()
        plt.savefig(f'Decreasing_{data_type}.png', dpi=300)
        plt.close()



    def Normalisation(self, data):
        """
        Normalising the csv items so that I can plot them on the same axis.
        """

        return np.array((data - data.min()) / (data.max() - data.min()))  # every key items will range from 0 to 1




if __name__=='__main__':
    Statistics(saving_csv=True)