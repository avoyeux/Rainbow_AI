"""
All the different models used in the deep learning tests using karine's masks. 
This was separated in another python file as it is still a work in progress.
"""

# Imports
from tensorflow import keras


class VaryingLSTMKernelSizes:
    """
    WORK IN PROGRESS
    Trying to see if I can improve the last architecture a little given my new understanding of the model.
    """

    def __init__(self, sequence_len: int = 8, image_size: int = 512, kernel_size_conv: list = [(3, 3), (5, 5), (7, 7)],
                 kernel_size_LSTM: list = [(9, 9), (5, 5), (3, 3)], first_filters: int = 128):

        # Arguments
        self.sequence_len = sequence_len  # number of images used for each sequences (i.e. the time dependence length)
        self.image_size = image_size  # pixel length of the post processed square images
        self.kernel_size_conv = kernel_size_conv  # kernel size for the Conv2D
        self.kernel_size_LSTM = kernel_size_LSTM  # kernel size for the ConvLSTM2D
        self.first_filters = first_filters  # initial number of filters in the U-Net
        
    def Down_block(self, x, filters: int, kernel_size: tuple, padding: str = 'same', strides: int = 1, first: bool = False):
        """
        To downscale the input and use an increasing number of filters.
        """

        kwargs = {'filters': filters, 'kernel_size': kernel_size, 'padding':padding, 'strides':strides, 
                  'activation': 'relu', 'return_sequences': True}

        if first:
            kwargs['input_shape'] = (self.sequence_len, self.image_size, self.image_size, 3)

        c = keras.layers.ConvLSTM2D(**kwargs)(x)
        p = keras.layers.TimeDistributed(keras.layers.MaxPool2D((2, 2), (2 ,2)))(c)
        return c, p

    def Up_block(self, x, skip, filters: int, kernel_size: tuple, padding: str = 'same', strides: int = 1):
        """
        To upscale the downscaled input.
        """

        kwargs = {'padding':padding, 'strides':strides, 'activation':'relu'}

        us = keras.layers.UpSampling2D((2, 2))(x)
        skip = keras.layers.Lambda(lambda x: x[:, -1, :, :, :])(skip)
        concat = keras.layers.Concatenate()([us, skip])
        c = keras.layers.Conv2D(filters, kernel_size, **kwargs)(concat)
        return c

    def Bottleneck(self, x, filters: int, kernel_size: tuple, padding: str = 'same', strides: int = 1):
        """
        Bridge between the downscaling and the upscaling.
        """
        
        kwargs = {'padding':padding, 'strides':strides, 'activation':'relu'}

        c = keras.layers.ConvLSTM2D(filters, self.kernel_size_LSTM[2], return_sequences=False, **kwargs)(x)
        return c

    def UNet(self):
        """
        Main structure of the model putting the downscaling, the bridge and the upscaling together.
        """

        f = [self.first_filters, self.first_filters * 2, self.first_filters * 4, self.first_filters * 8]
        inputs = keras.layers.Input((self.sequence_len, self.image_size, self.image_size, 3))

        p0 = inputs
        c1, p1 = self.Down_block(p0, f[0], kernel_size=self.kernel_size_LSTM[0], first=True)
        c2, p2 = self.Down_block(p1, f[1], kernel_size=self.kernel_size_LSTM[1])
        c3, p3 = self.Down_block(p2, f[2], kernel_size=self.kernel_size_LSTM[2])

        bn = self.Bottleneck(p3, f[3], kernel_size=(3, 3))

        u0 = self.Up_block(bn, c3, f[2], kernel_size=self.kernel_size_conv[0])
        u1 = self.Up_block(u0, c2, f[1], kernel_size=self.kernel_size_conv[1])
        u2 = self.Up_block(u1, c1, f[0], kernel_size=self.kernel_size_conv[2])

        outputs = keras.layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid', dtype='float32')(u2)
        model = keras.models.Model(inputs, outputs)
        return model
    
    def Model(self):
        """
        To have the same model function name for all models (easier to manipulate later on).
        """
        
        return self.UNet()

class VaryingLSTMKernelSizes_2Inputs:
    """
    WORK IN PROGRESS
    Trying to see if I can improve the last architecture a little given my new understanding of the model.
    """

    def __init__(self, sequence_len: int = 8, image_size: int = 512, kernel_size_conv: list = [(3, 3), (5, 5), (7, 7)],
                 kernel_size_LSTM: list = [(9, 9), (5, 5), (3, 3)], first_filters: int = 64):

        # Arguments
        self.sequence_len = sequence_len  # number of images used for each sequences (i.e. the time dependence length)
        self.image_size = image_size  # pixel length of the post processed square images
        self.kernel_size_conv = kernel_size_conv  # kernel size for the Conv2D
        self.kernel_size_LSTM = kernel_size_LSTM  # kernel size for the ConvLSTM2D
        self.first_filters = first_filters
        
    def Down_block(self, x, filters: int, kernel_size: tuple, padding: str = 'same', strides: int = 1, first: bool = False):
        """
        To downscale the input and use an increasing number of filters.
        """

        kwargs = {'filters': filters, 'kernel_size': kernel_size, 'padding':padding, 'strides':strides, 
                  'activation': 'relu', 'return_sequences': True}

        if first:
            kwargs['input_shape'] = (self.sequence_len, self.image_size, self.image_size, 2)

        c = keras.layers.ConvLSTM2D(**kwargs)(x)
        p = keras.layers.TimeDistributed(keras.layers.MaxPool2D((2, 2), (2 ,2)))(c)
        return c, p

    def Up_block(self, x, skip, filters: int, kernel_size: tuple, padding: str = 'same', strides: int = 1):
        """
        To upscale the downscaled input.
        """

        kwargs = {'padding':padding, 'strides':strides, 'activation':'relu'}

        us = keras.layers.UpSampling2D((2, 2))(x)
        skip = keras.layers.Lambda(lambda x: x[:, -1, :, :, :])(skip)
        concat = keras.layers.Concatenate()([us, skip])
        c = keras.layers.Conv2D(filters, kernel_size, **kwargs)(concat)
        return c

    def Bottleneck(self, x, filters: int, kernel_size: tuple, padding: str = 'same', strides: int = 1):
        """
        Bridge between the downscaling and the upscaling.
        """
        
        kwargs = {'padding':padding, 'strides':strides, 'activation':'relu'}

        c = keras.layers.ConvLSTM2D(filters, self.kernel_size_LSTM[2], return_sequences=False, **kwargs)(x)
        return c

    def UNet(self):
        """
        Main structure of the model putting the downscaling, the bridge and the upscaling together.
        """

        f = [self.first_filters, self.first_filters * 2, self.first_filters * 4, self.first_filters * 8]
        inputs = keras.layers.Input((self.sequence_len, self.image_size, self.image_size, 2))

        p0 = inputs
        c1, p1 = self.Down_block(p0, f[0], kernel_size=self.kernel_size_LSTM[0], first=True)
        c2, p2 = self.Down_block(p1, f[1], kernel_size=self.kernel_size_LSTM[1])
        c3, p3 = self.Down_block(p2, f[2], kernel_size=self.kernel_size_LSTM[2])

        bn = self.Bottleneck(p3, f[3], kernel_size=(3, 3))

        u0 = self.Up_block(bn, c3, f[2], kernel_size=self.kernel_size_conv[0])
        u1 = self.Up_block(u0, c2, f[1], kernel_size=self.kernel_size_conv[1])
        u2 = self.Up_block(u1, c1, f[0], kernel_size=self.kernel_size_conv[2])

        outputs = keras.layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid', dtype='float32')(u2)
        model = keras.models.Model(inputs, outputs)
        return model
    
    def Model(self):
        """
        To have the same model function name for all models (easier to manipulate later on).
        """
        
        return self.UNet()
