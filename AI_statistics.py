"""
To understand the statistics.csv file to hopefully get the best models.
"""

# Imports 
import os
import re
import glob 

import numpy as np
import pandas as pd


class Statistics:
    """
    To run some statistics on the statistic.csv file.
    """

    def __init__(self, saving_csv: bool = True):
        
        # Functions
        self.Paths()

        if saving_csv:
            self.CSV_to_pandas()

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

        train_time = self.Normalisation(df['Train loss'])
        train_acc = self.Normalisation(df['Train accuracy'])
        


    def Normalisation(self, data):
        """
        Normalising the csv items so that I can plot them on the same axis.
        """

        return (data - data.min()) / (data.max() - data.min())  # every key items will range from 0 to 1




if __name__=='__main__':
    Statistics(saving_csv=True)