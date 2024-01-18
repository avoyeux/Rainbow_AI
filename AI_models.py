"""
All the different models used in the deep learning tests.
"""

# Imports
from tensorflow import keras


class UNetModel:
    """
    First model and simplest model used.
    As the name implies, is follows a U-Net architecture by using CNNs, setting a number of filters for each convolutions and
    then downsampling and increasing the number of filters for the downsampled result. \
    
    Source: https://github.com/nikhilroxtomar/UNet-Segmentation-in-Keras-TensorFlow/blob/master/unet-segmentation.ipynb
    """

    def __init__(self, image_size: int = 512, kernel_size: tuple = (3, 3)):

        # Arguments
        self.image_size = image_size  # pixel length of the post processed square images
        self.kernel_size = kernel_size  # tuple representing the kernel size to be tested
        
    def Down_block(self, x, filters, padding='same', strides=1):
        """
        Down sampling block.
        Double convolution with a certain number of filters. The final result is then downsampled with a stride of 2.
        """

        c = keras.layers.Conv2D(filters, self.kernel_size, padding=padding, strides=strides, activation='relu')(x)
        c = keras.layers.Conv2D(filters, self.kernel_size, padding=padding, strides=strides, activation='relu')(c)
        p = keras.layers.MaxPool2D((2, 2), (2 ,2))(c)
        return c, p

    def Up_block(self, x, skip, filters, padding='same', strides=1):
        """
        Up sampling block.
        Upscales the input and does a double convolution, given a certain number of filters, on the result.
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
        Main structure of the U-Net architecture putting the downscaling, the bridge and the upscaling together.
        A small number of filters are used on the initial image and that number decreases while we downsample the image
        and increases when we upsample the result.
        """

        f = [32, 64, 128, 256, 512, 1024]  # number of filters used.
        inputs = keras.layers.Input((self.image_size, self.image_size, 2))  # represents the input shape 

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

        outputs = keras.layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(u4) #sigmoid as the output is a mask, i.e. 1 or 0
        model = keras.models.Model(inputs, outputs)
        return model
    
    def Model(self):
        """
        To have the same model function name for all models (easier to manipulate later on).
        """
        
        return self.UNet()


class UNetNDeepResidual:
    """
    Second model used. It is supposed to be an "improved" version of the usual U-Net architecture as deep residuals are also added in.
    That being said, in my case, when I was looking at the test results, I wasn't able to see any improvements. In some cases, the 
    results even seemed a little worse. Still, given the examples I had seen on internet, this method clearly has it's uses.
    
    Source: https://github.com/nikhilroxtomar/Deep-Residual-Unet/blob/master/Deep%20Residual%20UNet.ipynb 
    """

    def __init__(self, image_size: int = 512, kernel_size: tuple = (3, 3), padding: str = 'same', strides: int = 1):

        self.image_size = image_size  # pixel length of the post processed square images
        self.kernel_size = kernel_size  # tuple representing the kernel size to be tested
        self.padding = padding  # string that can be 'same' and 'valid'
        self.strides = strides  # the stride used, i.e. 1 is every pixel and 2 would create and output 2 times smaller for 1D. 
        
    def Bn_act(self, x, act=True):
        """
        Honestly, re-reading the code, I don't remember what this function actually does...
        From keras documentation, we get that:
        Batch normalization applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1.
        The layer will only normalize its inputs during inference after having been trained on data that has similar statistics 
        as the inference data.
        Furthermore, the activation layer just activates the 'relu' on the given layer used to introduce non-linearities into the network. 
        """

        x = keras.layers.BatchNormalization()(x)
        if act:
            x = keras.layers.Activation('relu')(x)
        return x
    
    def Conv_block(self, x, filters, strides=1):
        """
        Does a convolution and the bridge.
        """

        conv = self.Bn_act(x)
        conv = keras.layers.Conv2D(filters, self.kernel_size, padding=self.padding, strides=strides)(conv)
        return conv
    
    def Stem(self, x, filters):
        """
        Creates the initial convolution and the shortcut.
        """

        conv = keras.layers.Conv2D(filters, self.kernel_size, padding=self.padding, strides=self.strides)(x)
        conv = self.Conv_block(conv, filters)

        shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=self.padding, strides=self.strides)(x)
        shortcut = self.Bn_act(shortcut, act=False)

        output = keras.layers.Add()([conv, shortcut])
        return output
    
    def Residual_block(self, x, filters, strides=1):
        """
        Used in the downsampling (strides=2) and upsampling(strides=1). Does the convolutions and uses the batch normalisation.
        """

        res = self.Conv_block(x, filters, strides=strides)
        res = self.Conv_block(res, filters, strides=1)

        shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=self.padding, strides=strides)(x)
        shortcut = self.Bn_act(shortcut, act=False)

        output = keras.layers.Add()([shortcut, res])
        return output
    
    def Upsample_concat_block(self, x, xskip): 
        """
        The upsampling making the input 4 times bigger and using skip connections.
        """

        u = keras.layers.UpSampling2D((2, 2))(x)
        c = keras.layers.Concatenate()([u, xskip])
        return c

    def ResUNet(self):
        """
        Main structure of the deep residual U-Net architecture putting the downscaling, the bridge and the upscaling together.
        A small number of filters are used on the initial image and that number decreases while we downsample the image
        and increases when we upsample the result.
        """

        f = [32, 64, 128, 256, 512, 1024]  # number of filters
        inputs = keras.layers.Input((self.image_size, self.image_size, 2))  # represents the input shape 

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
    
    def Model(self):
        """
        To have the same model function name for all models (easier to manipulate later on).
        """

        return self.ResUNet()
    

class UNetConvLSTM2D_long:
    """
    U-Net architecture model using LSTM 2D convolutions for the downsampling, normal 2D convolutions
    for the upsampling with skip connections using the downsampling LSTM 2D convolutions to try and better 
    capture the local and global information. 
    The downscaling tries to capture the temporal evolution while the upscaling tries to capture the individual properties for
    the last image for which a mask is needed.
    """

    def __init__(self, sequence_len: int = 8, image_size: int = 512, kernel_size: tuple = (3, 3)):

        # Arguments
        self.sequence_len = sequence_len
        self.image_size = image_size  # pixel length of the post processed square images
        self.kernel_size = kernel_size  # tuple representing the kernel size to be tested
        
    def Down_block(self, x, filters, padding='same', strides=1, first=False):
        """
        To downscale the input and use an increasing number of filters.
        """

        kwargs = {'padding':padding, 'strides':strides, 'activation':'relu', 'return_sequences': True}
        if first:
            c = keras.layers.ConvLSTM2D(filters, self.kernel_size, 
                                        input_shape=(self.sequence_len, self.image_size, self.image_size, 3), **kwargs)(x)
        else:
            c = keras.layers.ConvLSTM2D(filters, self.kernel_size, **kwargs)(x)
        c = keras.layers.ConvLSTM2D(filters, self.kernel_size, **kwargs)(c)
        p = keras.layers.TimeDistributed(keras.layers.MaxPool2D((2, 2), (2 ,2)))(c)
        return c, p

    def Up_block(self, x, skip, filters, padding='same', strides=1):
        """
        To upscale the downscaled input.
        """

        kwargs = {'padding':padding, 'strides':strides, 'activation':'relu'}

        us = keras.layers.UpSampling2D((2, 2))(x)
        skip = keras.layers.Lambda(lambda x: x[:, -1, :, :, :])(skip)  # choosing the last image 
        concat = keras.layers.Concatenate()([us, skip])
        c = keras.layers.Conv2D(filters, self.kernel_size, **kwargs)(concat)
        c = keras.layers.Conv2D(filters, self.kernel_size, **kwargs)(c)
        return c

    def Bottleneck(self, x, filters, padding='same', strides=1):
        """
        Bridge between the downscaling and the upscaling.
        """

        kwargs = {'padding':padding, 'strides':strides, 'activation':'relu'}

        c = keras.layers.ConvLSTM2D(filters, self.kernel_size, return_sequences=True, **kwargs)(x)
        c = keras.layers.ConvLSTM2D(filters, self.kernel_size, return_sequences=False, **kwargs)(c)
        return c

    def UNet(self):
        """
        Main structure of the model putting the downscaling, the bridge and the upscaling together.
        """

        f = [32, 64, 128, 256, 512]  # number of filters
        inputs = keras.layers.Input((self.sequence_len, self.image_size, self.image_size, 3))  # represents the input shape 

        p0 = inputs
        c1, p1 = self.Down_block(p0, f[0], first=True)
        c2, p2 = self.Down_block(p1, f[1])
        c3, p3 = self.Down_block(p2, f[2])
        c4, p4 = self.Down_block(p3, f[3])

        bn = self.Bottleneck(p4, f[4])

        u1 = self.Up_block(bn, c4, f[3])
        u2 = self.Up_block(u1, c3, f[2])
        u3 = self.Up_block(u2, c2, f[1])
        u4 = self.Up_block(u3, c1, f[0])

        outputs = keras.layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(u4)
        model = keras.models.Model(inputs, outputs)
        return model
    
    def Model(self):
        """
        To have the same model function name for all models (easier to manipulate later on).
        """
        
        return self.UNet()


class UNetConvLSTM2D_short:
    """
    Created U-Net architecture model using LSTM 2D convolutions for the downsampling, normal 2D convolutions
    for the upsampling with skip connections using the downsampling LSTM 2D convolutions to try and better 
    capture the local and global information. 
    The difference with UNetConvLSTM2D_long is that only one convLSTM is done per downscaling and one Conv2D per upscaling.
    """

    def __init__(self, sequence_len: int = 8, image_size: int = 512, kernel_size: tuple = (3, 3)):

        # Arguments
        self.sequence_len = sequence_len  # number of images used for each sequences (i.e. the time dependence length)
        self.image_size = image_size  # pixel length of the post processed square images
        self.kernel_size = kernel_size  # tuple representing the kernel size to be tested
        
    def Down_block(self, x, filters, padding='same', strides=1, first=False):
        """
        To downscale the input and use an increasing number of filters.
        """

        kwargs = {'padding':padding, 'strides':strides, 'activation':'relu', 'return_sequences': True}

        if first:
            c = keras.layers.ConvLSTM2D(filters, self.kernel_size, 
                                        input_shape=(self.sequence_len, self.image_size, self.image_size, 3), **kwargs)(x)
        else:
            c = keras.layers.ConvLSTM2D(filters, self.kernel_size, **kwargs)(x)

        p = keras.layers.TimeDistributed(keras.layers.MaxPool2D((2, 2), (2 ,2)))(c)
        return c, p

    def Up_block(self, x, skip, filters, padding='same', strides=1):
        """
        To upscale the downscaled input.
        """

        kwargs = {'padding':padding, 'strides':strides, 'activation':'relu'}

        us = keras.layers.UpSampling2D((2, 2))(x)
        skip = keras.layers.Lambda(lambda x: x[:, -1, :, :, :])(skip)
        concat = keras.layers.Concatenate()([us, skip])
        c = keras.layers.Conv2D(filters, self.kernel_size, **kwargs)(concat)
        return c

    def Bottleneck(self, x, filters, padding='same', strides=1):
        """
        Bridge between the downscaling and the upscaling.
        """
        
        kwargs = {'padding':padding, 'strides':strides, 'activation':'relu'}

        c = keras.layers.ConvLSTM2D(filters, self.kernel_size, return_sequences=False, **kwargs)(x)
        return c

    def UNet(self):
        """
        Main structure of the model putting the downscaling, the bridge and the upscaling together.
        """

        f = [64, 128, 256, 512, 1024, 2048]
        inputs = keras.layers.Input((self.sequence_len, self.image_size, self.image_size, 3))

        p0 = inputs
        c1, p1 = self.Down_block(p0, f[0], first=True)
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
    
    def Model(self):
        """
        To have the same model function name for all models (easier to manipulate later on).
        """
        
        return self.UNet()
