# Discussion on the Models

This *Markdown* file was created to explain the model principle and the specific architecture and choices made.


### Summary:

- 1 - **The U-Net segmentation architecture**

- 2 - **Conv2D -- Capturing spatial features**

- 3 - **LSTM -- Capturing temporal features**

- 4 - **ConvLSTM2D -- Capturing spatio-temporal features**

- 5 - **U-Net ConvLSTM2D with Conv2D**

- 6 - **Code examples**


## 1 - The U-Net segmentation architecture

The *U-Net segmentation architecture* is mainly used in medical image analysis, biomedical research and was created for the aforementioned topic[1][UNet-wiki]. It was developed to try and overcome the challenges of limited labelled data[2][UNet-basics]. It is now also increasingly popular for satellite and aerial imagery analysis, in manufacturing process detection and autonomous driving[3][UNet-driving].

It is named as such because it follows a u-shaped structure. It initially does a convolution with a low number of filters on a high resolution image. From there, you downsample the result and increase the number of filters so that you have a better understanding of the larger features. At the bottom of the U, you have a bridge that is the convolution on the smallest downsampled result with the highest number of filters. You then do the same thing in reverse order, i.e. you upsample the result and do a convolution with a decreasing number of filters till you get an output that has the same size than the initial input. \
Furthermore, the upsampling path uses skip connections which directly concatenate features from the downsampling path to the corresponding upsampling one. While, even without the skip connections, the model will remember the weights gotten from the downsampling, this step is crucial as it combines the high-level abstract features with the low-level detailed ones gotten from the upsampling. Moreover, by bringing spatial details (e.g. textures, edges) directly to the upsampling it helps preserve what was lost due to repeated pooling. All in all, skip connections help in the combination of features, the preservation of spatial information, the improvement of localization accuracy and the mitigation of information loss. 


## 2 - Conv2D -- Capturing spatial features

As explained in the last section of this file, the *U-Net architecture* is a *Convolutional Neural Network* (CNN) as it uses convolutions to predict data labels. For 2D data (e.g. images), given a kernel size, you can use 2D convolutions that will help capture the spatial features of the input. To do so, a number of filters is chosen that, as the name implies, will each use a different set of values in the kernel. As such, it will create an output that detects different features. \
In the case of model training, after one batch of inputs is processed, the initial random filter values are updated to minimise the loss. This is done by backpropagation of the loss calculated when comparing the final output and the true labels of the original input. 

Other important choices that can be made when using a 2D convolution are the *padding*, the *stride* and the *activation method*: 
- **padding:** refers to the adding of extra pixels around the edges of the input so that the kernel can also process the edges of the input. As such, two choices are generally possible: *valid* and *same*. *Valid* meaning that no padding is used, and so the edges of the input are not processed and the output is smaller than the input. *Same* meaning that extra pixels are added so that the edges are processed giving to the output the same shape than the original input (if the *stride* is equal to 1).
- **strides:** refers to the step size by which the filter is moved. Hence, increasing the strides value (integer or tuple for 2D images), will downsample the input and decrease the output size. E.g. A strides of (2, 2) for a 2D image, with *'same'* padding, will give a feature output that is 4 times smaller than the input.
- **activation method:** this is specific to deep learning frameworks. It represents a non-linear transformation used to introduce non-linearity in the outputs so that the neural network doesn't collapse to a linear transformation. The network only doing linear transformations, like convolutions or multiplication, means that regardless of the number of layers, the end result will be the same than only one, more complex, linear transformation. This implies that without an *activation function*, the network cannot learn non-linear relations even though it is generally the case in real word scenarios. In this project, two different activation methods where used: *'ReLu'* and *'sigmoig'*. *ReLu* returns the standard max(x, 0) value while  *sigmoid* returns 1 / (1 + exp(-x)), i.e. small values are close to 0 and high values close to 1 (it is mainly used to create masks)[4][conv2D-activation].

In *Keras.Conv2D*, a lot of other arguments are possible, but in this specific project, they weren't used[5][Conv2D-doc].


## - 3 - LSTM -- Capturing temporal features[6][LSTM-basics]

The *Long short-term memory* (LSTM) network is a type of *recurrent neural network* (RNN) and as such is used to learn temporal dependencies. 

In a basic RNN, the information gotten from treating the first input of the ordered sequence will be passed to the second step which will be treating the second input. Then, the knowledge gotten from the manipulation of the second input (which also has information from the first treatment) is passed on and etc. That being said, quite quickly, the information from the furthest predictions is lost, i.e. simple RNNs have only a short-term memory. This is where LSTMs come into play.

LSTMs are, as the name implies, a method in which the short-term capabilities of RNNs are saved in a 'separate' memory to then be used in the following computations. While there are a multitude of different LSTMs, they are generally made up of a cell (the 'separate' memory) and three types of gates which act with the cell; the *forget gate*, the *input gate* and the *output gate*. The *forget gate* decides what information needs to be discarded from the cell. The *input gate* decides what should be added to the cell. Lastly, the *output gate* decides what part of the cell should be used in the current step. 


## 4 - ConvLSTM2D -- Capturing spatio-temporal features[7][ConvLSTM2D-paper]

As the name implies, the *convLSTM2D* method is the fusion of a convolution with the architecture of an LSTM. \
It is important to note that this process is different from using an LSTM structure on a 1D dense array gotten from a convolution operation of a an image (that would correspond to a CNN-LSTM model[8][CNN-LSTM-doc] also called a FC-LSTM with convolutional layers[7][ConvLSTM2D-paper]). 

The convLSTM2D method keeps a better understanding of the evolution of the spatial properties though time as the dimension of the input are kept during the LSTM. To do so, the convolution is done by the LSTM gates themselves, i.e. the LSTM cell will itself be a 2D structures and the gates will also make decisions based on the spatial properties of the input as they act as a 2D convolution operation[7][ConvLSTM2D-paper]. Therefore, this computation is done in one layer and processes the spatio-temporal properties of the sequence, as opposed to two layers that process the spatial features and then the temporal dependencies of those spatial features (i.e. CNN-LSTM like models). 

Given the difference between how the convolution is done in a CNN-LSTM model and a convolution LSTM model, the effect of the kernel size is, likewise, different. In a CNN-RNN model, the increase in kernel size will permit the convolution to capture broader spatial features (i.e. a larger 'object') and the LSTM side will look at the temporal dependencies of those broad features. In a convolutional LSTM, the kernel size will directly influence how spatial changes are integrated overtime. Hence, an increase in kernel size will detect more significant spatial changes over time, i.e. it will capture faster motions as opposed to just bigger objects changing[7][ConvLSTM2D-paper].  

In the tensorflow.keras.layers.ConvLSTM2D[9][ConvLSTM2D-doc] there is an argument used that I still haven't introduced; the *return_sequences* argument. It gives you the choice to return the whole sequence or only the last output of the sequence. There are a lot of arguments that I haven't discussed here as I haven't used nor researched them yet...


## 5 - U-Net ConvLSTM2D with Conv2D

This particular architecture is the most complex model I have used.\
In this project, the input used was made up of an ordered sequence of 3 2-dimensional grayscale (i.e. 1 channel) matrices. The first one is an untreated *STEREO* acquisition. The second, a treated one. The third, the corresponding mask (i.e. label) for these images. The last 3 2-dimensional matrices of the sequence has an empty array (i.e. full of zeros) as a placeholder for the mask. The expected output of the model is the mask corresponding to that placeholder.

By separating the u-shaped architecture in 3 parts, we have the left arc, the base curve and the right arc.

- **The left arc** (i.e. the downsampling)**:** the downsampling of this architecture is made up of convolutional 2-dimensional LSTM functions (*ConvLSTM2D*). This was done to study the spatio-temporal aspect of the movement of the solar protuberance. The thinking was that, as the protuberance can only be pinpointed by inference when looking at an ordered sequence of images, then the model needs to be able to understand spatio-temporal dependencies. 

- **The base curve** (i.e. the bridge)**:** the bridge is made up of convolutional 2 dimensional LSTM functions but end result will only be the last output of the sequence. In my case, it is the results from the 2 *STEREO* images and the placeholder. The output represents only the last sequence output as the next part of the architecture will be simple 2-dimensional convolutions.

- **The right arc** (i.e. the upsampling)**:** the upsampling is made up of 2-dimensional convolutions with skip connections. These convolutions are done on the results for the last image. This is because the mask of this particular image is what needs to be predicted. As such, the skip connections used are the results gotten from the last image of the corresponding downsampling path. The thinking was that, as the protuberance position highly depends on the spatial features of the last image, the mask is created from the spatial features of the final image 'only'.

In summary, this model was created to try and spatio-temporally understand the high-level abstract features using *ConvLSTM2D* functions and then study the low-level detailed spatial features using the *Conv2D* function.\
Furthermore, the skip connections, used to try and recover the spatial features lost due to repeated pooling, will also keep spatio-temporal information as it is taken from the convLSTM2D functions.

Given my still really limited knowledge, understanding and experience in deep learning, the model used is not the best choice for the task at hand. After further thinking and research, the use of CNN-RNN model, or a *Conv2D* then *ConvLSTM2D* model might give more desirable results as the spatial properties of the inputs are the main factor in labelling the noisy images. The temporal features are only important to localise the general position of the protuberance. \
Moreover, the kernel size for the *ConvLSTM2D* and the *Conv2D* doesn't need to be the same as they don't have the same role. I should probably even use varying kernel sizes for each downsampling or upsampling path depending on the general shape and movements of the protuberance in the corresponding path scale. 


## 6 - Code examples

This part is to showcase some of the examples related to *U-Net architectures* and *convolutional LSTM models* that I was able to find. 

- [1][UNet-first]: this is the first (ever) model that I have used. It is basically the same model than the first one used in this project. Furthermore, a corresponding *Youtube* video also exists[10][UNet-firstvid].
- [2][UNet-second]: this is the second one which is the *deep residual U-Net* architecture. This code also has a corresponding *Youtube* video[11][UNet-secondvid].
- [3][example-3]: a simple example of using *ConvLSTM2D* functions in deep learning.
- [4][example-4]: a full example of a U-Net architecture using *Conv2D* in the downsampling and a mix of *Conv2D* and *ConvLSTM2D* in the upscaling. Will most likely use this code example to improve my code.

There is also a paper that explains their method used for precipitation nowcasting that I am planning to test[12][convLSTM2D-superbpaper].

[UNet-basics]: https://www.geeksforgeeks.org/u-net-architecture-explained/ 
[UNet-wiki]: https://en.wikipedia.org/wiki/U-Net
[UNet-driving]: https://arxiv.org/abs/2110.04079v5
[Conv2D-activation]: https://keras.io/api/layers/activations/
[Conv2D-doc]: https://keras.io/api/layers/convolution_layers/convolution2d/
[LSTM-basics]: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
[ConvLSTM2D-paper]: https://arxiv.org/abs/1506.04214v2
[CNN-LSTM-doc]: https://www.nature.com/articles/s41598-023-41314-y
[ConvLSTM2D-doc]: https://keras.io/api/layers/recurrent_layers/conv_lstm2d/
[UNet-first]: https://github.com/nikhilroxtomar/UNet-Segmentation-in-Keras-TensorFlow/blob/master/unet-segmentation.ipynb
[UNet-firstvid]: https://www.youtube.com/watch?v=M3EZS__Z_XE
[UNet-second]: https://github.com/nikhilroxtomar/Deep-Residual-Unet/blob/master/Deep%20Residual%20UNet.ipynb
[UNet-secondvid]: https://www.youtube.com/watch?v=BOoBWRTpaKk
[example-3]: https://www.kaggle.com/code/kcostya/convlstm-convolutional-lstm-network-tutorial#:~:text=URL%3A%20https%3A%2F%2Fwww.kaggle.com%2Fcode%2Fkcostya%2Fconvlstm
[example-4]: https://github.com/rezazad68/BCDU-Net/blob/master/Lung%20Segmentation/models.py
[ConvLSTM2D-superbpaper]: https://www.researchgate.net/publication/377335518_Enhancing_Radar_Echo_Extrapolation_by_ConvLSTM2D_for_Precipitation_Nowcasting