"""
To plot the test data next to the corresponding model prediction.
"""

# Imports 
import os
import numpy as np
import matplotlib.pyplot as plt

from common_alf import PlotFunctions as Plot


class Plotter:
    """
    Plots the test and prediction for the test given the corresponding data and path.
    """

    def __init__(self, path : str, test_images: np.ndarray, test_masks: np.ndarray, predicted_masks: np.ndarray, image_size: int = 512):

        # Arguments
        self.path = path
        self.test_images = test_images
        self.test_masks = test_masks
        self.predicted_masks = predicted_masks
        self.image_size = image_size

        # Functions
        self.Plot()

    def Plot(self):
        """
        Plots the prediction and the actual masks for the testing set.
        """

        fig = plt.figure(figsize=(20, 15))
        a = 0
        for loop in range(4):
            for loop2 in range(5):
                a += 1
                ax = fig.add_subplot(4, 5, a)
                if loop2 == 0:
                    ax.imshow(np.reshape(self.test_masks[loop] * 255, (self.image_size, self.image_size)), interpolation='none', cmap='gray')
                elif loop2 == 1:
                    ax.imshow(np.reshape(self.test_images[loop, :, :, 0] * 255, (self.image_size, self.image_size)), interpolation='none', cmap='gray')
                elif loop2 == 2:
                    ax.imshow(np.reshape(self.test_images[loop, :, :, 1] * 255, (self.image_size, self.image_size)), interpolation='none', cmap='gray')
                elif loop2 == 3:
                    ax.imshow(np.reshape(self.predicted_masks[loop] * 255, (self.image_size, self.image_size)), interpolation='none', cmap='gray')
                else:
                    ax.imshow(np.reshape(self.test_images[loop, :, :, 0] * 255, (self.image_size, self.image_size)), interpolation='none', cmap='gray')
                    lines = Plot.Contours(np.reshape(self.test_masks[loop], (self.image_size, self.image_size)))
                    for line in lines:
                        ax.plot(line[1], line[0], color='b', linewidth=0.5, alpha=0.3)
                    lines = Plot.Contours(np.reshape(self.predicted_masks[loop], (self.image_size, self.image_size)))
                    for line in lines:
                        ax.plot(line[1], line[0], color='r', linewidth=0.5, alpha=0.3)
        plt.savefig(os.path.join(self.path, 'full_plot.png'), dpi=500)
        plt.close()

        fig = plt.figure(figsize=(20, 30))
        a = 0
        for loop in range(4):
            for loop2 in range(3):
                a += 1
                ax = fig.add_subplot(4, 3, a)
                if loop2 == 0:
                    ax.imshow(np.reshape(self.test_masks[loop] * 255, (self.image_size, self.image_size)), interpolation='none', cmap='gray')
                elif loop2 == 1:
                    ax.imshow(np.reshape(self.predicted_masks[loop] * 255, (self.image_size, self.image_size)), interpolation='none', cmap='gray')
                else:
                    ax.imshow(np.reshape(self.test_masks[loop] * 255, (self.image_size, self.image_size)), interpolation='none',  cmap='gray')
                    lines = Plot.Contours(np.reshape(self.predicted_masks[loop], (self.image_size, self.image_size)))
                    for line in lines:
                        ax.plot(line[1], line[0], color='r', linewidth=0.5, alpha=0.3)
        plt.savefig(os.path.join(self.path, 'only_masks.png'), dpi=300)
        plt.close()


class Plotter_LSTM:
    """
    Plots the test and prediction for the LSTM test given the corresponding data and path.
    """

    def __init__(self, path: str, test_images: np.ndarray, test_masks: np.ndarray, predicted_masks: np.array, image_size: int = 512):

        # Arguments
        self.path = path
        self.test_images = test_images
        self.test_masks = test_masks
        self.predicted_masks = predicted_masks
        self.image_size = image_size

        # Functions
        self.Plot()

    def Plot(self):
        """
        Plots the prediction and the actual masks for the testing set.
        """

        fig = plt.figure(figsize=(20, 15))
        a = 0
        for loop in range(3):
            for loop2 in range(5):
                a += 1
                ax = fig.add_subplot(3, 5, a)
                if loop2 == 0:
                    ax.imshow(np.reshape(self.test_masks[loop] * 255, (self.image_size, self.image_size)), interpolation='none', cmap='gray')
                elif loop2 == 1:
                    ax.imshow(np.reshape(self.test_images[loop, -1, :, :, 0] * 255, (self.image_size, self.image_size)), interpolation='none', cmap='gray')
                elif loop2 == 2:
                    ax.imshow(np.reshape(self.test_images[loop, -1, :, :, 1] * 255, (self.image_size, self.image_size)), interpolation='none', cmap='gray')
                elif loop2 == 3:
                    ax.imshow(np.reshape(self.predicted_masks[loop] * 255, (self.image_size, self.image_size)), interpolation='none', cmap='gray')
                else:
                    ax.imshow(np.reshape(self.test_images[loop, -1, :, :, 0] * 255, (self.image_size, self.image_size)), interpolation='none', cmap='gray')
                    lines = Plot.Contours(np.reshape(self.test_masks[loop], (self.image_size, self.image_size)))
                    for line in lines:
                        ax.plot(line[1], line[0], color='b', linewidth=0.5, alpha=0.3)
                    lines = Plot.Contours(np.reshape(self.predicted_masks[loop], (self.image_size, self.image_size)))
                    for line in lines:
                        ax.plot(line[1], line[0], color='r', linewidth=0.5, alpha=0.3)
        plt.savefig(os.path.join(self.path, 'full_plot.png'), dpi=500)
        plt.close()

        fig = plt.figure(figsize=(20, 30))
        a = 0
        for loop in range(3):
            for loop2 in range(3):
                a += 1
                ax = fig.add_subplot(3, 3, a)
                if loop2 == 0:
                    ax.imshow(np.reshape(self.test_masks[loop] * 255, (self.image_size, self.image_size)), interpolation='none', cmap='gray')
                elif loop2 == 1:
                    ax.imshow(np.reshape(self.predicted_masks[loop] * 255, (self.image_size, self.image_size)), interpolation='none', cmap='gray')
                else:
                    ax.imshow(np.reshape(self.test_masks[loop] * 255, (self.image_size, self.image_size)), interpolation='none',  cmap='gray')
                    lines = Plot.Contours(np.reshape(self.predicted_masks[loop], (self.image_size, self.image_size)))
                    for line in lines:
                        ax.plot(line[1], line[0], color='r', linewidth=0.5, alpha=0.3)
        plt.savefig(os.path.join(self.path, 'only_masks.png'), dpi=300)
        plt.close()