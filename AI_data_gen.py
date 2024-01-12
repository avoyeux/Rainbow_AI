"""
Where the data is preprocessed to then be used when fitting the deep learning models.
"""

# Imports
import os

import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.utils import shuffle

class DataGen:
    """
    To find the data and create the training inputs and outputs and the test set for the deep learning.
    """

    def __init__(self, image_size=512, mask_val=0, test_length=4, augmentation=True):
        # Inputs
        self.image_size = image_size
        self.mask_val = mask_val
        self.test_length = int(test_length)
        self.augmentation = augmentation

        # Arguments
        self.Paths()
        self.on_epoch_end()

    def Paths(self):
        """
        Creating the paths dictionary.
        """
        
        main_path = '/home/avoyeux/AI_tests'

        self.paths = {'Main': main_path,
                      'Inputs': os.path.join(main_path, 'Inputs'),
                      'Inputs2': os.path.join(main_path, 'Inputs2'),
                      'Masks': os.path.join(main_path, 'New_masks'),}

        self.input_paths = sorted(Path(self.paths['Inputs']).glob('*.png')) 
        self.input_paths2 = sorted(Path(self.paths['Inputs2']).glob('*.png'))
        self.output_paths = sorted(Path(self.paths['Masks']).glob('*.png'))

        # Tensorflow doesn't like pathlib so there you go
        self.input_paths = [str(path) for path in self.input_paths]
        self.input_paths2 = [str(path) for path in self.input_paths2]
        self.output_paths = [str(path) for path in self.output_paths]

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
        Loading and initial manipulation on the ouputs, i.e. masks.
        """

        mask = tf.io.read_file(path)
        mask = tf.image.decode_png(mask, channels=3)
        mask = tf.image.rgb_to_grayscale(mask)
        mask = tf.image.resize(mask, [self.image_size, self.image_size])
        mask = tf.cast(mask, tf.float32) / 255
        mask = np.array(mask)
        mask[mask < 1] = 0
        return mask

    def Data_augmentation(self, input_data):
        """
        Flipping and rotating the data so that the model is less prone to overfitting (hopefully).
        """

        img1 = np.flip(input_data, axis=0)
        img2 = np.flip(input_data, axis=1)
        img3 = np.rot90(input_data)
        img4 = np.rot90(input_data, k=2)
        img5 = np.rot90(input_data, k=3)
        return img1, img2, img3, img4, img5

    def Give_data(self):
        """
        Getting all the masks and images to then be used in the model for training and testing.
        """

        train_images = []
        train_masks = []
        for number, input_path in enumerate(self.input_paths[:-self.test_length]):
            _img1 = self.Load_inputs(input_path)
            _img2 = self.Load_inputs(self.input_paths2[number])
            _mask = self.Load_outputs(self.output_paths[number])
            _imgtot = np.concatenate((_img1, _img2), axis=2)

            if self.augmentation:
                nw_img1, nw_img2, nw_img3, nw_img4, nw_img5 = self.Data_augmentation(_imgtot)
                nw_mask1, nw_mask2, nw_mask3, nw_mask4, nw_mask5 = self.Data_augmentation(_mask)
                train_images.extend([_imgtot, nw_img1, nw_img2, nw_img3, nw_img4, nw_img5])
                train_masks.extend([_mask, nw_mask1, nw_mask2, nw_mask3, nw_mask4, nw_mask5])
            else:
                train_images.append(_imgtot)
                train_masks.append(_mask)
        train_images = np.array(train_images)
        train_masks = np.array(train_masks)

        test_images = []
        test_masks = []
        for number, input_path in enumerate(self.input_paths[-self.test_length:]):
            number += len(self.input_paths[:-self.test_length]) 

            _img1 = self.Load_inputs(input_path)
            _img2 = self.Load_inputs(self.input_paths2[number])
            _mask = self.Load_outputs(self.output_paths[number])
            _imgtot = np.concatenate((_img1, _img2), axis=2)
            test_images.append(_imgtot)
            test_masks.append(_mask)
        test_images = np.array(test_images)
        test_masks = np.array(test_masks)
        return train_images, train_masks, test_images, test_masks

    def on_epoch_end(self):
        """
        No clue what this does but in the video I was looking at for Unet architectures, this was done. 
        """

        pass


class DataGen_ordered:
    """
    To find the data and create the training inputs and outputs and the test set for the deep learning.
    """

    def __init__(self, image_size=512, mask_val=0, sequence_len=8):
        # Inputs
        self.image_size = image_size
        self.mask_val = mask_val
        self.sequence_len = sequence_len

        # Arguments
        self.Paths()
        self.Separated_paths()
        self.Sliding_window_paths()

    def Paths(self):
        """
        Creating the paths dictionary.
        """
        
        main_path = '/home/avoyeux/AI_tests'

        self.paths = {'Main': main_path,
                      'Inputs': os.path.join(main_path, 'Inputs'),
                      'Inputs2': os.path.join(main_path, 'Inputs2'),
                      'Masks': os.path.join(main_path, 'New_masks')}

        self.input_paths = sorted(Path(self.paths['Inputs']).glob('*.png'))
        self.input_paths2 = sorted(Path(self.paths['Inputs2']).glob('*.png'))
        self.output_paths = sorted(Path(self.paths['Masks']).glob('*.png'))

        # Tensorflow doesn't like pathlib so there you go
        self.input_paths = [str(path) for path in self.input_paths]
        self.input_paths2 = [str(path) for path in self.input_paths2]
        self.output_paths = [str(path) for path in self.output_paths]
    
    def Separated_paths(self):
        """
        To separate the paths in sections with images that follow each other to then be able to use it in an LSTM model.
        This is only done as my initial set of images is made up of 3 different sections and is not continuous.
        """

        paths_sections = []
        paths_sections.append(self.Section_path(0, 20))  # image nb85 to 104, keeping 1 for tests
        paths_sections.append(self.Section_path(21, 51))  # image nb215 to 244, keeping 1 for tests
        paths_sections.append(self.Section_path(52, 77))  # image nb345 to 369, keeping 1 for tests

        self.paths_sections = paths_sections

    def Section_path(self, nb_init, nb_end):
        """
        For the Separated_paths function.
        """
        
        section =[]
        for loop in range(nb_init, nb_end):
            section.append([self.input_paths[loop], self.input_paths2[loop], self.output_paths[loop]])
        return np.array(section)
    
    def Sliding_window_paths(self):
        """
        Uses the ordered path sections to then separate them into sliding sequences to use in a model using LSTM. 
        """

        sliding_sequences = []
        for section in self.paths_sections:
            for loop in range(len(section) - self.sequence_len + 1):
                slide = section[loop:loop + self.sequence_len] 
                sliding_sequences.append(slide)
        
        self.sliding_sequences = np.array(sliding_sequences)
                

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
    
    def Test_data(self):
        """
        To define the test set.
        """

        indexes = [20, 51, 77]
        test_input = []
        test_output = []
        for index in indexes:
            test_sequence = []
            for loop in range((index + 1) - self.sequence_len, index + 1):
                _img1 = self.Load_inputs(self.input_paths[loop])
                _img2 = self.Load_inputs(self.input_paths2[loop])
                _mask = self.Load_outputs(self.output_paths[loop])
                _input = np.concatenate([_img1, _img2, _mask], axis=2)
                test_sequence.append(_input)
            test_sequence = np.array(test_sequence)
            mask = np.copy(test_sequence[-1, :, :, -1])
            test_output.append(mask)
            test_sequence[-1, :, :, -1] = np.zeros((self.image_size, self.image_size), dtype='float32')
            test_input.append(test_sequence)
        return np.array(test_input), np.array(test_output)
        
    def Give_data(self):
        """
        Creates the train sets for a sequential method (like LSTM).
        """

        train_inputs = []
        train_outputs = []

        for sequence in self.sliding_sequences:
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
            nw_sq1, nw_sq2, nw_sq3, nw_sq4, nw_sq5 = self.Data_augmentation(sequence_inputs, dimensions=4)
            nw_msk1, nw_msk2, nw_msk3, nw_msk4, nw_msk5 = self.Data_augmentation(output, dimensions=3)

            train_inputs.extend([sequence_inputs, nw_sq1, nw_sq2, nw_sq3, nw_sq4, nw_sq5])
            train_outputs.extend([output, nw_msk1, nw_msk2, nw_msk3, nw_msk4, nw_msk5])
        
        train_inputs = np.array(train_inputs)
        train_outputs = np.array(train_outputs)

        train_inputs, train_outputs = shuffle(train_inputs, train_outputs)
        test_input, test_output = self.Test_data()
        return train_inputs, train_outputs, test_input, test_output


if __name__=='__main__':
    test = DataGen_ordered()
    train_inputs, train_outputs, test_input, test_output = test.Give_data()
    print(f'train input shape is {train_inputs.shape} with size {train_inputs.nbytes / 1024**2}MB')
    print(f"if float16 then {(train_inputs.astype('float16').nbytes)}")
    print(f'train output shape is {train_outputs.shape} with size {train_outputs.nbytes / 1024**2}MB')
    print(f'test input shape is {test_input.shape} with size {test_input.nbytes / 1024**2}MB')
    print(f'test output shape is {test_output.shape}with size {test_output.nbytes / 1024**2}MB')

    test_input0 = test_input[0]
    print(f'test input0 shape is {test_input0.shape}')
    
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(test_output[0], interpolation='none')
    plt.colorbar()
    plt.savefig(os.path.join(os.getcwd(), f'FUCK1.png'), dpi=300)
    plt.close()
    print(f'test_output0.shape is {test_output[0].shape}')
