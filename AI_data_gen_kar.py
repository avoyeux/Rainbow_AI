"""
Where the data is preprocessed to then be used when fitting the deep learning models.
This code is only created to preprocess the Karine's data to know which model is best for the task at hand. 
"""

# Imports
import re
import os

import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.utils import shuffle



class KarineDataGenOrdered:
    """
    To find the data and create the training inputs and outputs and the test set for the deep learning.
    """

    def __init__(self, image_size: int = 512, sequence_len: int = 8):

        # Inputs
        self.image_size = image_size  # the final square image size in pixels (the initial image is resized to this value)
        self.sequence_len = sequence_len  # number of images used for each sequences (i.e. the time dependence length)

        # Created attributes
        self.input_paths = None  # sorted list of filepath strings for the non treated intensity STEREO images
        self.input_paths2 = None  # same for the treated average STEREO images
        self.output_paths = None  # same for the corresponding masks
        self.paths_sections = None  # inhomogeneous list with "shape" (3, nb_of_training_images, 3) with the filepath string for each index of the three above attributes
        self.sliding_sequences = None  # array with shape (nb_of_training_images, sequence_length, 3) with the filepath strings

        # Functions
        self.Paths()
        self.Patterns()
        self.Continuous_sequences()
        self.Validation_set()
        self.Testing_set()

    def Paths(self):
        """
        Creating the paths dictionary.
        """
        
        main_path = os.getcwd()
        paths = {'Main': main_path,
                      'Inputs': os.path.join(main_path, 'Inputs_kar'),
                      'Inputs2': os.path.join(main_path, 'Inputs2_kar'),
                      'Masks': os.path.join(main_path, 'masque_karine_processed')}

        self.input_paths = Path(paths['Inputs']).glob('*.png')
        self.input_paths2 = Path(paths['Inputs2']).glob('*.png')
        self.output_paths = Path(paths['Masks']).glob('*.png')

        # Tensorflow doesn't like pathlib objects
        self.input_paths = sorted([str(path) for path in self.input_paths])
        self.input_paths2 = sorted([str(path) for path in self.input_paths2])
        self.output_paths = sorted([str(path) for path in self.output_paths])
    
    def Patterns(self):
        """
        To get the image numbers. 
        """

        mask_pattern = re.compile(r'frame(\d{4})\.png')
        self.numbers = [int(mask_pattern.match(os.path.basename(path)).group(1)) for path in self.output_paths]

    def Continuous_sequences(self):
        """
        To separate the paths in sections with images that follow each other to then be able to use it in an LSTM model.
        As the initial set of images is not continuous, this pre-processing is needed.
        """

        sequences = []
        start_index = 0

        for i in range(1, len(self.numbers)):
            if self.numbers[i] != self.numbers[i - 1] + 1:
                sequences.append((start_index, i - 1))
                start_index = i
        
        # Last sequence
        sequences.append((start_index, len(self.numbers) - 1))
        self.sequences_indexes = sequences
        
        paths_sections = []
        for tuples in sequences:
            paths_sections.append(self.Section_path(tuples[0], tuples[1] - 6))  # -6 to keep the last images for the validation and testing set
        self.sliding_sequences = self.Sliding_window_paths(paths_sections)  # the training sliding sequences

    def Section_path(self, nb_init, nb_end):
        """
        For the Separated_paths function.
        """
        
        section = []
        for loop in range(nb_init, nb_end + 1):
            section.append([self.input_paths[loop], self.input_paths2[loop], self.output_paths[loop]])
        return np.array(section)
    
    def Sliding_window_paths(self, paths_sections):
        """
        Uses the ordered path sections to then separate them into sliding sequences to use in a model using LSTM. 
        """

        sliding_sequences = []
        for section in paths_sections:
            for loop in range(len(section) - self.sequence_len + 1):
                slide = section[loop:loop + self.sequence_len] 
                sliding_sequences.append(slide)
        return np.array(sliding_sequences)

    def Validation_set(self):
        """
        Creating the validation set.
        It is made up of the 5 before last sliding window sequences of each continuous set so that the last masks haven't been used at all in the training. 
        All in all, this corresponds to 35 sequences, i.e. about 14.6% of the initial data (with 239 labelled images).
        """
        
        paths_sections = []
        for tuples in self.sequences_indexes:
            start_index = tuples[1] - self.sequence_len - 4  # -4 as it is -5 + 2 - 1 (hard to explain but it makes sense if you think about it)
            paths_sections.append(self.Section_path(start_index, tuples[1] - 1))  # -1 for the testing set
        self.validation_sliding_sequences = self.Sliding_window_paths(paths_sections)

    def Testing_set(self):
        """
        Creating the testing set.
        Composed of the last sliding window sequences of each continuous set.
        All in all, 7 sliding window sequences are therefore used for the testing set.
        This represents about 2.9% of the data.
        """
        
        paths_sections = []
        for tuples in self.sequences_indexes:
            start_index = tuples[1] - self.sequence_len + 1
            paths_sections.append(self.Section_path(start_index, tuples[1]))
        self.testing_sliding_sequences = self.Sliding_window_paths(paths_sections)

    def Load_inputs(self, path):
        """
        Loading and initial manipulation on the input images.
        """

        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.resize(image, [self.image_size, self.image_size])
        image = tf.cast(image, tf.float32) / 255
        return np.array(image)

    def Load_outputs(self, path):
        """
        Loading and initial manipulation on the outputs, i.e. masks.
        """

        mask = tf.io.read_file(path)
        mask = tf.image.decode_png(mask, channels=3)
        mask = tf.image.rgb_to_grayscale(mask)
        mask = tf.image.resize(mask, [self.image_size, self.image_size])
        mask = tf.cast(mask, tf.float32) / 255
        mask = np.array(mask)
        mask[mask < 1] = 0
        return mask

    def Data_augmentation(self, input_data, dimensions):
        """
        Flipping and rotating the data so that the model is less prone to overfitting (hopefully).
        """

        img1 = np.flip(input_data, axis=dimensions - 3)
        img2 = np.flip(input_data, axis=dimensions - 2)
        img3 = np.rot90(input_data, axes=(dimensions-3, dimensions-2))
        img4 = np.rot90(input_data, axes=(dimensions-3, dimensions-2), k=2)
        img5 = np.rot90(input_data, axes=(dimensions-3, dimensions-2), k=3)
        return img1, img2, img3, img4, img5
    
    def Give_data_sets(self, sequences_list: list, augmentation: bool = False):
        """
        Creates the sets given the paths to the images.
        It is used for the training, validation and testing sets.
        """

        inputs = []
        outputs = []

        for sequence in sequences_list:
            output = self.Load_outputs(sequence[-1, -1])
            sequence[-1, -1] = 'none'  # Taking away the last mask as it is the output

            sequence_inputs = []
            for paths in sequence:
                _img1 = self.Load_inputs(paths[0])
                _img2 = self.Load_inputs(paths[1])
                if paths[2]=='none':
                    _mask = np.zeros((self.image_size, self.image_size, 1), dtype='float32')
                else:
                    _mask = self.Load_outputs(paths[2])
                _input_tot = np.concatenate([_img1, _img2, _mask], axis=2)
                sequence_inputs.append(_input_tot)

            sequence_inputs = np.array(sequence_inputs)
            if augmentation:
                nw_sq1, nw_sq2, nw_sq3, nw_sq4, nw_sq5 = self.Data_augmentation(sequence_inputs, dimensions=4)
                nw_msk1, nw_msk2, nw_msk3, nw_msk4, nw_msk5 = self.Data_augmentation(output, dimensions=3)

                inputs.extend([sequence_inputs, nw_sq1, nw_sq2, nw_sq3, nw_sq4, nw_sq5])
                outputs.extend([output, nw_msk1, nw_msk2, nw_msk3, nw_msk4, nw_msk5])
            else:
                inputs.append(sequence_inputs)
                outputs.append(output)

        inputs = np.array(inputs)
        outputs = np.array(outputs)

        inputs, outputs = shuffle(inputs, outputs)
        return inputs, outputs
    
    def Give_data(self):
        """
        Returns all the different sets for the training, validation and testing.
        """

        training_inputs, training_outputs = self.Give_data_sets(self.sliding_sequences, augmentation=True)
        validation_inputs, validation_outputs = self.Give_data_sets(self.validation_sliding_sequences)
        testing_inputs, testing_outputs = self.Give_data_sets(self.testing_sliding_sequences)
        return training_inputs, training_outputs, validation_inputs, validation_outputs, testing_inputs, testing_outputs
    
class KarineDataGenOrdered_2Inputs:
    """
    To find the data and create the training inputs and outputs and the test set for the deep learning.
    """

    def __init__(self, image_size: int = 512, sequence_len: int = 8):

        # Inputs
        self.image_size = image_size  # the final square image size in pixels (the initial image is resized to this value)
        self.sequence_len = sequence_len  # number of images used for each sequences (i.e. the time dependence length)

        # Created attributes
        self.input_paths2 = None  # same for the treated intensity STEREO images
        self.output_paths = None  # same for the corresponding masks
        self.paths_sections = None  # inhomogeneous list with "shape" (3, nb_of_training_images, 3) with the filepath string for each index of the three above attributes
        self.sliding_sequences = None  # array with shape (nb_of_training_images, sequence_length, 3) with the filepath strings

        # Functions
        self.Paths()
        self.Patterns()
        self.Continuous_sequences()
        self.Validation_set()
        self.Testing_set()

    def Paths(self):
        """
        Creating the paths dictionary.
        """
        
        main_path = os.getcwd()
        paths = {'Main': main_path,
                      'Inputs2': os.path.join(main_path, 'Inputs2_kar'),
                      'Masks': os.path.join(main_path, 'masque_karine_processed')}

        self.input_paths2 = Path(paths['Inputs2']).glob('*.png')
        self.output_paths = Path(paths['Masks']).glob('*.png')

        # Tensorflow doesn't like pathlib objects
        self.input_paths2 = sorted([str(path) for path in self.input_paths2])
        self.output_paths = sorted([str(path) for path in self.output_paths])
    
    def Patterns(self):
        """
        To get the image numbers. 
        """

        mask_pattern = re.compile(r'frame(\d{4})\.png')
        self.numbers = [int(mask_pattern.match(os.path.basename(path)).group(1)) for path in self.output_paths]

    def Continuous_sequences(self):
        """
        To separate the paths in sections with images that follow each other to then be able to use it in an LSTM model.
        As the initial set of images is not continuous, this pre-processing is needed.
        """

        sequences = []
        start_index = 0

        for i in range(1, len(self.numbers)):
            if self.numbers[i] != self.numbers[i - 1] + 1:
                sequences.append((start_index, i - 1))
                start_index = i
        
        # Last sequence
        sequences.append((start_index, len(self.numbers) - 1))
        self.sequences_indexes = sequences
        
        paths_sections = []
        for tuples in sequences:
            paths_sections.append(self.Section_path(tuples[0], tuples[1] - 6))  # -6 to keep the last images for the validation and testing set
        self.sliding_sequences = self.Sliding_window_paths(paths_sections)  # the training sliding sequences

    def Section_path(self, nb_init, nb_end):
        """
        For the Separated_paths function.
        """
        
        section = []
        for loop in range(nb_init, nb_end + 1):
            section.append([self.input_paths2[loop], self.output_paths[loop]])
        return np.array(section)
    
    def Sliding_window_paths(self, paths_sections):
        """
        Uses the ordered path sections to then separate them into sliding sequences to use in a model using LSTM. 
        """

        sliding_sequences = []
        for section in paths_sections:
            for loop in range(len(section) - self.sequence_len + 1):
                slide = section[loop:loop + self.sequence_len] 
                sliding_sequences.append(slide)
        return np.array(sliding_sequences)

    def Validation_set(self):
        """
        Creating the validation set.
        It is made up of the 5 before last sliding window sequences of each continuous set so that the last masks haven't been used at all in the training. 
        All in all, this corresponds to 35 sequences, i.e. about 14.6% of the initial data (with 239 labelled images).
        """
        
        paths_sections = []
        for tuples in self.sequences_indexes:
            start_index = tuples[1] - self.sequence_len - 4  # -4 as it is -5 + 2 - 1 (hard to explain but it makes sense if you think about it)
            paths_sections.append(self.Section_path(start_index, tuples[1] - 1))  # -1 for the testing set
        self.validation_sliding_sequences = self.Sliding_window_paths(paths_sections)

    def Testing_set(self):
        """
        Creating the testing set.
        Composed of the last sliding window sequences of each continuous set.
        All in all, 7 sliding window sequences are therefore used for the testing set.
        This represents about 2.9% of the data.
        """
        
        paths_sections = []
        for tuples in self.sequences_indexes:
            start_index = tuples[1] - self.sequence_len + 1
            paths_sections.append(self.Section_path(start_index, tuples[1]))
        self.testing_sliding_sequences = self.Sliding_window_paths(paths_sections)

    def Load_inputs(self, path):
        """
        Loading and initial manipulation on the input images.
        """

        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.resize(image, [self.image_size, self.image_size])
        image = tf.cast(image, tf.float32) / 255
        return np.array(image)

    def Load_outputs(self, path):
        """
        Loading and initial manipulation on the outputs, i.e. masks.
        """

        mask = tf.io.read_file(path)
        mask = tf.image.decode_png(mask, channels=3)
        mask = tf.image.rgb_to_grayscale(mask)
        mask = tf.image.resize(mask, [self.image_size, self.image_size])
        mask = tf.cast(mask, tf.float32) / 255
        mask = np.array(mask)
        mask[mask < 1] = 0
        return mask

    def Data_augmentation(self, input_data, dimensions):
        """
        Flipping and rotating the data so that the model is less prone to overfitting (hopefully).
        """

        img1 = np.flip(input_data, axis=dimensions - 3)
        img2 = np.flip(input_data, axis=dimensions - 2)
        img3 = np.rot90(input_data, axes=(dimensions-3, dimensions-2))
        img4 = np.rot90(input_data, axes=(dimensions-3, dimensions-2), k=2)
        img5 = np.rot90(input_data, axes=(dimensions-3, dimensions-2), k=3)
        return img1, img2, img3, img4, img5
    
    def Give_data_sets(self, sequences_list: list, augmentation: bool = False):
        """
        Creates the sets given the paths to the images.
        It is used for the training, validation and testing sets.
        """

        inputs = []
        outputs = []

        for sequence in sequences_list:
            output = self.Load_outputs(sequence[-1, -1])
            sequence[-1, -1] = 'none'  # Taking away the last mask as it is the output

            sequence_inputs = []
            for paths in sequence:
                _img2 = self.Load_inputs(paths[0])
                if paths[1]=='none':
                    _mask = np.zeros((self.image_size, self.image_size, 1), dtype='float32')
                else:
                    _mask = self.Load_outputs(paths[1])
                _input_tot = np.concatenate([_img2, _mask], axis=2)
                sequence_inputs.append(_input_tot)

            sequence_inputs = np.array(sequence_inputs)
            if augmentation:
                nw_sq1, nw_sq2, nw_sq3, nw_sq4, nw_sq5 = self.Data_augmentation(sequence_inputs, dimensions=4)
                nw_msk1, nw_msk2, nw_msk3, nw_msk4, nw_msk5 = self.Data_augmentation(output, dimensions=3)

                inputs.extend([sequence_inputs, nw_sq1, nw_sq2, nw_sq3, nw_sq4, nw_sq5])
                outputs.extend([output, nw_msk1, nw_msk2, nw_msk3, nw_msk4, nw_msk5])
            else:
                inputs.append(sequence_inputs)
                outputs.append(output)

        inputs = np.array(inputs)
        outputs = np.array(outputs)

        inputs, outputs = shuffle(inputs, outputs)
        return inputs, outputs
    
    def Give_data(self):
        """
        Returns all the different sets for the training, validation and testing.
        """

        training_inputs, training_outputs = self.Give_data_sets(self.sliding_sequences, augmentation=True)
        validation_inputs, validation_outputs = self.Give_data_sets(self.validation_sliding_sequences)
        testing_inputs, testing_outputs = self.Give_data_sets(self.testing_sliding_sequences)
        return training_inputs, training_outputs, validation_inputs, validation_outputs, testing_inputs, testing_outputs


class KarineDataGenOrdered_1Input:
    """
    To find the data and create the training inputs and outputs and the test set for the deep learning.
    """

    def __init__(self, image_size: int = 512, sequence_len: int = 8):

        # Inputs
        self.image_size = image_size  # the final square image size in pixels (the initial image is resized to this value)
        self.sequence_len = sequence_len  # number of images used for each sequences (i.e. the time dependence length)

        # Created attributes
        self.input_paths2 = None  # sorted list of filepath strings for the average (treated) STEREO images
        self.output_paths = None  # same for the corresponding masks
        self.paths_sections = None  # inhomogeneous list with "shape" (3, nb_of_training_images, 3) with the filepath string for each index of the three above attributes
        self.sliding_sequences = None  # array with shape (nb_of_training_images, sequence_length, 3) with the filepath strings

        # Functions
        self.Paths()
        self.Patterns()
        self.Continuous_sequences()
        self.Validation_set()
        self.Testing_set()

    def Paths(self):
        """
        Creating the paths dictionary.
        """
        
        main_path = os.getcwd()
        paths = {'Main': main_path,
                      'Inputs2': os.path.join(main_path, 'Inputs2_kar'),
                      'Masks': os.path.join(main_path, 'masque_karine_processed')}

        self.input_paths2 = Path(paths['Inputs2']).glob('*.png')
        self.output_paths = Path(paths['Masks']).glob('*.png')

        # Tensorflow doesn't like pathlib objects
        self.input_paths2 = sorted([str(path) for path in self.input_paths2])
        self.output_paths = sorted([str(path) for path in self.output_paths])
    
    def Patterns(self):
        """
        To get the image numbers. 
        """

        mask_pattern = re.compile(r'frame(\d{4})\.png')
        self.numbers = [int(mask_pattern.match(os.path.basename(path)).group(1)) for path in self.output_paths]

    def Continuous_sequences(self):
        """
        To separate the paths in sections with images that follow each other to then be able to use it in an LSTM model.
        As the initial set of images is not continuous, this pre-processing is needed.
        """

        sequences = []
        start_index = 0

        for i in range(1, len(self.numbers)):
            if self.numbers[i] != self.numbers[i - 1] + 1:
                sequences.append((start_index, i - 1))
                start_index = i
        
        # Last sequence
        sequences.append((start_index, len(self.numbers) - 1))
        self.sequences_indexes = sequences
        
        paths_sections = []
        for tuples in sequences:
            paths_sections.append(self.Section_path(tuples[0], tuples[1] - 6))  # -6 to keep the last images for the validation and testing set
        self.sliding_sequences = self.Sliding_window_paths(paths_sections)  # the training sliding sequences

    def Section_path(self, nb_init, nb_end):
        """
        For the Separated_paths function.
        """
        
        section = []
        for loop in range(nb_init, nb_end + 1):
            section.append([self.input_paths2[loop], self.output_paths[loop]])
        return np.array(section)
    
    def Sliding_window_paths(self, paths_sections):
        """
        Uses the ordered path sections to then separate them into sliding sequences to use in a model using LSTM. 
        """

        sliding_sequences = []
        for section in paths_sections:
            for loop in range(len(section) - self.sequence_len + 1):
                slide = section[loop:loop + self.sequence_len] 
                sliding_sequences.append(slide)
        return np.array(sliding_sequences)

    def Validation_set(self):
        """
        Creating the validation set.
        It is made up of the 5 before last sliding window sequences of each continuous set so that the last masks haven't been used at all in the training. 
        All in all, this corresponds to 35 sequences, i.e. about 14.6% of the initial data (with 239 labelled images).
        """
        
        paths_sections = []
        for tuples in self.sequences_indexes:
            start_index = tuples[1] - self.sequence_len - 4  # -4 as it is -5 + 2 - 1 (hard to explain but it makes sense if you think about it)
            paths_sections.append(self.Section_path(start_index, tuples[1] - 1))  # -1 for the testing set
        self.validation_sliding_sequences = self.Sliding_window_paths(paths_sections)

    def Testing_set(self):
        """
        Creating the testing set.
        Composed of the last sliding window sequences of each continuous set.
        All in all, 7 sliding window sequences are therefore used for the testing set.
        This represents about 2.9% of the data.
        """
        
        paths_sections = []
        for tuples in self.sequences_indexes:
            start_index = tuples[1] - self.sequence_len + 1
            paths_sections.append(self.Section_path(start_index, tuples[1]))
        self.testing_sliding_sequences = self.Sliding_window_paths(paths_sections)

    def Load_inputs(self, path):
        """
        Loading and initial manipulation on the input images.
        """

        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.resize(image, [self.image_size, self.image_size])
        image = tf.cast(image, tf.float32) / 255
        return np.array(image)

    def Load_outputs(self, path):
        """
        Loading and initial manipulation on the outputs, i.e. masks.
        """

        mask = tf.io.read_file(path)
        mask = tf.image.decode_png(mask, channels=3)
        mask = tf.image.rgb_to_grayscale(mask)
        mask = tf.image.resize(mask, [self.image_size, self.image_size])
        mask = tf.cast(mask, tf.float32) / 255
        mask = np.array(mask)
        mask[mask < 1] = 0
        return mask

    def Data_augmentation(self, input_data, dimensions):
        """
        Flipping and rotating the data so that the model is less prone to overfitting (hopefully).
        """

        img1 = np.flip(input_data, axis=dimensions - 3)
        img2 = np.flip(input_data, axis=dimensions - 2)
        img3 = np.rot90(input_data, axes=(dimensions-3, dimensions-2))
        img4 = np.rot90(input_data, axes=(dimensions-3, dimensions-2), k=2)
        img5 = np.rot90(input_data, axes=(dimensions-3, dimensions-2), k=3)
        return img1, img2, img3, img4, img5
    
    def Give_data_sets(self, sequences_list: list, augmentation: bool = False):
        """
        Creates the sets given the paths to the images.
        It is used for the training, validation and testing sets.
        """

        inputs = []
        for sequence in sequences_list:

            sequence_inputs = []
            for paths in sequence:
                _img2 = self.Load_inputs(paths[0])
                _mask = self.Load_outputs(paths[1])
                _input_tot = np.concatenate([_img2, _mask], axis=2)
                sequence_inputs.append(_input_tot)

            sequence_inputs = np.array(sequence_inputs)
            if augmentation:
                nw_sq1, nw_sq2, nw_sq3, nw_sq4, nw_sq5 = self.Data_augmentation(sequence_inputs, dimensions=4)

                inputs.extend([sequence_inputs, nw_sq1, nw_sq2, nw_sq3, nw_sq4, nw_sq5])
            else:
                inputs.append(sequence_inputs)

        results = np.array(inputs)
        inputs = results[:, :, :, :, 0]
        outputs = results[:, :, :, :, 1]
        inputs = np.expand_dims(inputs, axis=-1)
        outputs = np.expand_dims(outputs, axis=-1)
        inputs, outputs = shuffle(inputs, outputs)
        return inputs, outputs
    
    def Give_data(self):
        """
        Returns all the different sets for the training, validation and testing.
        """

        training_inputs, training_outputs = self.Give_data_sets(self.sliding_sequences, augmentation=True)
        validation_inputs, validation_outputs = self.Give_data_sets(self.validation_sliding_sequences)
        testing_inputs, testing_outputs = self.Give_data_sets(self.testing_sliding_sequences)
        return training_inputs, training_outputs, validation_inputs, validation_outputs, testing_inputs, testing_outputs
    

    
if __name__=='__main__':
    test = KarineDataGenOrdered_2Inputs()
    data = test.Give_data()
    print(f'testing shapes are {data[0].shape} and {data[1].shape}')
    print(f'validation shapes are {data[2].shape} and {data[3].shape}')
    print(f'testing shapes are {data[4].shape} and {data[5].shape}')