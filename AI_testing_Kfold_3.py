import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from tensorflow import keras



class DataGen(keras.utils.Sequence):
    """
    To find the data and create the corresponding inputs and outputs for the deep learning.
    """

    def __init__(self, image_size=256, mask_val=1, augmentation=True):
        # Inputs
        self.image_size = image_size
        self.mask_val = mask_val
        self.augmentation = augmentation

        # Arguments
        self.Paths()
        self.Initial_manipulation()
        self.on_epoch_end()

    def Paths(self):
        """
        Creating the paths dictionary.
        """
        
        main_path = '/home/avoyeux/AI_tests'

        self.paths = {'Main': main_path,
                      'Inputs': os.path.join(main_path, 'Inputs'),
                      'Inputs2': os.path.join(main_path, 'Inputs2'),
                      'Masks': os.path.join(main_path, 'New_masks'), 
                      'Test': os.path.join(main_path, 'test')}
        os.makedirs(self.paths['Test'], exist_ok=True)

        self.all_input_paths = sorted(Path(self.paths['Inputs']).glob('*.png'))
        self.all_input_paths2 = sorted(Path(self.paths['Inputs2']).glob('*.png'))
        self.all_output_paths = sorted(Path(self.paths['Masks']).glob('*.png'))

        # Tensorflow doesn't like pathlib so there you go
        self.input_paths = [str(path) for path in self.all_input_paths[:-4]]
        self.input_paths2 = [str(path) for path in self.all_input_paths2[:-4]]
        self.output_paths = [str(path) for path in self.all_output_paths[:-4]]

    def Initial_manipulation(self):
        """
        Adding an initial mask representing the general position of the output mask"""

        image = tf.io.read_file(os.path.join(self.paths['Main'], 'masks_sum_colors.png'))
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, [self.image_size, self.image_size])

        target_color = np.array([38, 255, 255])
        self.initial_filter = np.all(image == target_color, axis=-1)

    def Load_inputs(self, path):
        """
        Loading and initial manipulation on the input images.
        """

        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.resize(image, [self.image_size, self.image_size])
        image = tf.cast(image, tf.float32) / 255
        image = np.array(image)
        image[self.initial_filter] = self.mask_val
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
        mask[self.initial_filter] = 0
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
        Getting all the masks and images to then be used in the model.
        """

        images = []
        masks = []
        for number, input_path in enumerate(self.input_paths):
            _img1 = self.Load_inputs(input_path)
            _img2 = self.Load_inputs(self.input_paths2[number])
            _mask = self.Load_outputs(self.output_paths[number])
            _imgtot = np.concatenate((_img1, _img2), axis=2)
            if self.augmentation:
                nw_img1, nw_img2, nw_img3, nw_img4, nw_img5 = self.Data_augmentation(_imgtot)
                nw_mask1, nw_mask2, nw_mask3, nw_mask4, nw_mask5 = self.Data_augmentation(_mask)
                images.extend([_imgtot, nw_img1, nw_img2, nw_img3, nw_img4, nw_img5])
                masks.extend([_mask, nw_mask1, nw_mask2, nw_mask3, nw_mask4, nw_mask5])
            else:
                images.append(_imgtot)
                masks.append(_mask)
        images = np.array(images)
        masks = np.array(masks)
        return images, masks

    def on_epoch_end(self):
        pass


class UNetModel(DataGen):
    """
    First model and simplest model used.
    """

    def __init__(self, image_size=256, kernel_size=(3, 3)):
        super().__init__(image_size=image_size)
        self.image_size = image_size
        self.kernel_size = kernel_size

        # The model
        self.model = self.UNet()
        
    def Down_block(self, x, filters, padding='same', strides=1):
        """
        To downscale the input.
        """

        c = keras.layers.Conv2D(filters, self.kernel_size, padding=padding, strides=strides, activation='relu')(x)
        c = keras.layers.Conv2D(filters, self.kernel_size, padding=padding, strides=strides, activation='relu')(c)
        p = keras.layers.MaxPool2D((2, 2), (2 ,2))(c)
        return c, p

    def Up_block(self, x, skip, filters, padding='same', strides=1):
        """
        To upscale the downscaled input.
        """

        us = keras.layers.UpSampling2D((2, 2))(x)
        concat = keras.layers.Concatenate()([us, skip])
        c = keras.layers.Conv2D(filters, self.kernel_size, padding=padding, strides=strides, activation='relu')(concat)
        c = keras.layers.Conv2D(filters, self.kernel_size, padding=padding, strides=strides, activation='relu')(c)
        return c

    def Bottleneck(self, x, filters, padding='same', strides=1):
        """
        Bridge between the downscaling and the upscaling.
        """

        c = keras.layers.Conv2D(filters, self.kernel_size, padding=padding, strides=strides, activation='relu')(x)
        c = keras.layers.Conv2D(filters, self.kernel_size, padding=padding, strides=strides, activation='relu')(c)
        return c

    def UNet(self):
        """
        Main structure  of the UNet model putting the downscaling, the bridge and the upscaling together.
        """

        f = [16, 32, 64, 128, 256, 512]
        inputs = keras.layers.Input((self.image_size, self.image_size, 2))

        p0 = inputs
        c1, p1 = self.Down_block(p0, f[0])
        c2, p2 = self.Down_block(p1, f[1])
        c3, p3 = self.Down_block(p2, f[2])
        c4, p4 = self.Down_block(p3, f[3])
        c5, p5 = self.Down_block(p4, f[4])

        bn = self.Bottleneck(p5, f[5])

        u0 = self.Up_block(bn, c5, f[4])
        u1 = self.Up_block(u0, c4, f[3])
        u2 = self.Up_block(u1, c3, f[2])
        u3 = self.Up_block(u2, c2, f[1])
        u4 = self.Up_block(u3, c1, f[0])

        outputs = keras.layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(u4)
        model = keras.models.Model(inputs, outputs)
        return model


class UNetNDeepResidual(DataGen):
    """
    Second model used. More complex than the first one as it also has deep residuals embedded inside it.
    """

    def __init__(self, image_size=256, kernel_size=(3, 3), padding='same', strides=1):
        super().__init__(image_size=image_size)
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides

        # The model
        self.model = self.ResUNet()
        
    def Bn_act(self, x, act=True):

        x = keras.layers.BatchNormalization()(x)
        if act:
            x = keras.layers.Activation('relu')(x)
        return x
    
    def Conv_block(self, x, filters, strides=1):
        """
        Creates the bridge between the downscaling and the upscaling.
        """

        conv = self.Bn_act(x)
        conv = keras.layers.Conv2D(filters, self.kernel_size, padding=self.padding, strides=strides)(conv)
        return conv
    
    def Stem(self, x, filters):

        conv = keras.layers.Conv2D(filters, self.kernel_size, padding=self.padding, strides=self.strides)(x)
        conv = self.Conv_block(conv, filters)

        shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=self.padding, strides=self.strides)(x)
        shortcut = self.Bn_act(shortcut, act=False)

        output = keras.layers.Add()([conv, shortcut])
        return output
    
    def Residual_block(self, x, filters, strides=1):

        res = self.Conv_block(x, filters, strides=strides)
        res = self.Conv_block(res, filters, strides=1)

        shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=self.padding, strides=strides)(x)
        shortcut = self.Bn_act(shortcut, act=False)

        output = keras.layers.Add()([shortcut, res])
        return output
    
    def Upsample_concat_block(self, x, xskip): 

        u = keras.layers.UpSampling2D((2, 2))(x)
        c = keras.layers.Concatenate()([u, xskip])
        return c

    def ResUNet(self):

        f = [16, 32, 64, 128, 256, 512]
        inputs = keras.layers.Input((self.image_size, self.image_size, 2))

        # Encoder
        e0 = inputs
        e1 = self.Stem(e0, f[0])
        e2 = self.Residual_block(e1, f[1], strides=2)
        e3 = self.Residual_block(e2, f[2], strides=2)
        e4 = self.Residual_block(e3, f[3], strides=2)
        e5 = self.Residual_block(e4, f[4], strides=2)
        e6 = self.Residual_block(e5, f[5], strides=2)

        # Bridge
        b0 = self.Conv_block(e6, f[5])
        b1 = self.Conv_block(b0, f[5])

        # Decoder
        u1 = self.Upsample_concat_block(b1, e5)
        d1 = self.Residual_block(u1, f[5])

        u2 = self.Upsample_concat_block(d1, e4)
        d2 = self.Residual_block(u2, f[4])

        u3 = self.Upsample_concat_block(d2, e3)
        d3 = self.Residual_block(u3, f[3])

        u4 = self.Upsample_concat_block(d3, e2)
        d4 = self.Residual_block(u4, f[2])

        u5 = self.Upsample_concat_block(d4, e1)
        d5 = self.Residual_block(u5, f[1])

        outputs = keras.layers.Conv2D(1, (1, 1), padding=self.padding, activation='sigmoid')(d5)
        model = keras.models.Model(inputs, outputs)
        return model


class RunningTheModel:
    """
    To run the deep learning given a chosen model.
    """

    def __init__(self, epochs=200, **kwargs):
        # Model class attributes
        super().__init__(**kwargs)
        self.model_name = self.__class__.__bases__[1].__name__
        self.epochs = epochs

        # Functions
        self.Updated_paths()
        self.Compiling_the_model()
        self.Predict_n_plot()

    def Updated_paths(self):
        """
        Creates the subpaths to save the models and the test visualisations.
        """

        self.paths['Model_folder'] = os.path.join(self.paths['Main'], self.model_name)
        self.paths['Kernel_size'] = os.path.join(self.paths['Model_folder'], f'kernel{self.kernel_size[0]}_{self.kernel_size[1]}')
        self.paths['Epochs'] = os.path.join(self.paths['Kernel_size'], f'Epochs{self.epochs}')  

        os.makedirs(self.paths['Epochs'], exist_ok=True)

    def Compiling_the_model(self):
        """
        Compiles the model and saves it.
        """

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        data = DataGen()
        x, y = data.Give_data()

        self.model.fit(x, y, epochs=self.epochs)
        self.model.save_weights(os.path.join(self.paths['Epochs'], 'model.h5'))

    def Importing_tests(self):
        """
        Imports the testing set not used in the model training
        """

        test_img_paths = [str(path) for path in self.all_input_paths[-4:]]
        test_img_paths2 = [str(path) for path in self.all_input_paths2[-4:]]
        test_mask_paths = [str(path) for path in self.all_output_paths[-4:]]

        images = []
        masks = []
        for number, test_img in enumerate(test_img_paths):
            _img = self.Load_inputs(test_img)
            _img2 = self.Load_inputs(test_img_paths2[number])
            _mask = self.Load_outputs(test_mask_paths[number])

            _imgtot = np.concatenate((_img, _img2), axis=2)
            images.append(_imgtot)
            masks.append(_mask)

        images = np.array(images)
        masks = np.array(masks)
        return images, masks
    
    def Predict_n_plot(self):
        """
        Plots the prediction and the actual masks for the testing set.
        """

        images, masks = self.Importing_tests()
        results = self.model.predict(images)
        results2 = results > 0.1

        fig = plt.figure(figsize=(20, 15))

        a = 0
        for loop in range(4):
            for loop2 in range(5):
                a += 1
                ax = fig.add_subplot(4, 5, a)
                if loop2 == 0:
                    ax.imshow(np.reshape(masks[loop] * 255, (256, 256)), interpolation='none', cmap='gray')
                elif loop2 == 1:
                    ax.imshow(np.reshape(images[loop, :, :, 0] * 255, (256, 256)), interpolation='none', cmap='gray')
                elif loop2 == 2:
                    ax.imshow(np.reshape(images[loop, :, :, 1] * 255, (256, 256)), interpolation='none', cmap='gray')
                elif loop2 == 3:
                    ax.imshow(np.reshape(results2[loop] * 255, (256, 256)), interpolation='none', cmap='gray')
                else:
                    ax.imshow(np.reshape(images[loop, :, :, 0] * 255, (256, 256)), interpolation='none', cmap='gray')
                    lines = self.Contours(np.reshape(masks[loop], (256,256)))
                    for line in lines:
                        ax.plot(line[1], line[0], color='b', linewidth=0.5, alpha=0.3)
                    lines = self.Contours(np.reshape(results2[loop], (256,256)))
                    for line in lines:
                        ax.plot(line[1], line[0], color='r', linewidth=0.5, alpha=0.3)
        plt.savefig(os.path.join(self.paths['Epochs'], 'full_plot.png'), dpi=400)
        plt.close()

        a = 0
        fig = plt.figure(figsize=(20, 30))
        for loop in range(4):
            for loop2 in range(3):
                a += 1
                ax = fig.add_subplot(4, 3, a)
                if loop2 == 0:
                    ax.imshow(np.reshape(masks[loop] * 255, (256, 256)), interpolation='none', cmap='gray')
                elif loop2 == 1:
                    ax.imshow(np.reshape(results2[loop] * 255, (256, 256)), interpolation='none', cmap='gray')
                else:
                    ax.imshow(np.reshape(masks[loop] * 255, (256, 256)), interpolation='none',  cmap='gray')
                    lines = self.Contours(np.reshape(results2[loop], (256, 256)))
                    for line in lines:
                        ax.plot(line[1], line[0], color='r', linewidth=0.5, alpha=0.3)
        plt.savefig(os.path.join(self.paths['Epochs'], 'only_masks.png'), dpi=200)
        plt.close()

    @staticmethod
    def Contours(mask):
        """
        To plot the contours given a mask
        Source: https://stackoverflow.com/questions/40892203/can-matplotlib-contours-match-pixel-edges
        """

        pad = np.pad(mask, [(1, 1), (1, 1)])  # zero padding
        im0 = np.abs(np.diff(pad, n=1, axis=0))[:, 1:]
        im1 = np.abs(np.diff(pad, n=1, axis=1))[1:, :]
        lines = []
        for ii, jj in np.ndindex(im0.shape):
            if im0[ii, jj] == 1:
                lines += [([ii - .5, ii - .5], [jj - .5, jj + .5])]
            if im1[ii, jj] == 1:
                lines += [([ii - .5, ii + .5], [jj - .5, jj - .5])]
        return lines


def Child_class_factory(ParentClass):
    """
    To choose which parent class (i.e. model) is used when running the code (i.e. the RunningTheModel class).
    """
    
    class DynamicChild(RunningTheModel, ParentClass):
        def __init__(self, epochs=200, **kwargs):
            super().__init__(epochs=epochs, **kwargs)
    return DynamicChild


if __name__ == '__main__':

    # Running the code for different models
    kernel_sizes = [(3, 3), (5, 5)]
    epochs = [10, 50, 100, 200]

    for kernel_size in kernel_sizes:
        for epoch in epochs:
            DynamicChildA = Child_class_factory(UNetModel)
            instance_a = DynamicChildA(epochs=epoch, kernel_size=kernel_size)

            DynamicChildB = Child_class_factory(UNetNDeepResidual)
            instance_b = DynamicChildB(epochs=epoch, kernel_size=kernel_size)