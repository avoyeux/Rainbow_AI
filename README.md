# CNN and RNN (LSTM) codes for UNet image segmentation tasks:

### Summary of the README:

- 1 - The github repo: explanation of the uses of the different Python codes in the repository.

- 2 - Malpractices: explanation of the main unethical decisions I have made.

- 3 - Changes: major changes that can, and most probably should, be done.


## 1 - The github repo:

- `AI_code.py`: Python code encompassing all the other python codes. It is where I choose the different arguments, models and results I want to plot.

- `AI_code_LSTM.py`: Same than AI_code.py. The only difference is that, given the input and output being different between the normal CNN model and the LSTM RNN model, the functions are a little bit different. In essence though, it is the same python file.

- `AI_data_gen.py`: Python file to find, preprocess and output the training and testing data sets for the AI models. Two different classes are present as one is for the initial CNN models and the second ordered one is for the LSTM models.

- `AI_models.py`: Python file where all the different AI models used are saved. 

- `AI_plotting.py`: Python file where the plotting class is stored to then plot the testing set results.

- `common_alf.py`: Python file where some usual functions, that I regularly, use are stored. As of now, it is still pretty empty, i.e. still a work in progress. 

- `requirements.txt`: Text file with all the dependencies needed to run the code.


## 2 - Malpractices:

While the AI models I have used are relevant, due to the small size of the labelled data, I have made some important unethical decisions.

These wrongdoings are:
- **Data sets:** It is imperative to have a training, validation and testing data set. The training data set (usually 60-80% of the labelled data) is used to train the model, the validation set to fine tune the weights and the testing set to test it on 'not seen' labelled data. In my case, I have only used a training and a really small testing data set. When having an extremely small labelled data size, you should use a k-fold(*) with only 1 labelled image for the validation set. I didn't use it here as I was testing a lot of different parameters for different models (it would have been too computationally expensive as I still wasn't sure which method to fine tune).

- **Epochs** (i.e. a complete pass through the entire training set)**:** When choosing the number of epochs needed when training the model, the way to do so should be to stop the training when the validation loss(**) starts to increase (clear sign of overfitting). 


> **\*k-fold cross-validation:** a loop that separates the labelled data in k sections so that all data is at least once used in the training set and in the validation set. This technique is used to evaluate the performance of a given model on the data. It is hence used for hyperparameter tuning as it outputs the average performance (e.g. the accuracy). That being said, it is really computationally expensive as you have to train the model as many times as there are folds. For time series, *time series cross-validation* is needed.

>****validation loss:** When training a model with a training and validation set, the training loss is the "error" when training the model and the validation loss is the "error" when validating the model. Hence, the training loss nearly always decreases (can have some small increase, but it should decrease or stagnate in the short/long term). If the validation loss starts increasing, then the training is overfitting. 


## 3 - Changes:

Because I am still new to coding and it was my first time trying deep learning models, a lot of major changes are recommended.

- Given the recent improvements made in the PyTorch library, Keras is less and less used for computationally expensive deep learning tasks. That being said, Keras provides a higher-level API and, as such, demands way less time to produce a functioning model. On the other hand, using PyTorch is essential for creating more complex models.

- GPU: Be it PyTorch or Keras, both have optimisation capabilities by leveraging the computational properties of GPUs (going from a 30% increase in efficiency to 10-50 times the initial efficiency for large datasets using complex models).

- Transformers: In 2017, the Transformer model was introduced in a paper titled "Attention is All You Need" by Ashish Vaswani et al. This model also processes sequential data (like in RNNs, e.g. LSTM) but is way more efficient as it processes the entire sequence simultaneously and handles long-term dependencies better. That being said, it is worse on understanding and preserving sequential information, i.e. transformers are better for understanding dependencies over long-term periods while LSTM are better for short-term dependencies. Furthermore, and being the reasons why I didn't use it, transformers need a large data set to not overfit and the model architecture is way more complex.


### <div align="center"> <font color="red"> Disclaimer </font> </div>

<div align="center">
My understanding is still really limited and so, what I have done an explained here might be inaccurate. 
I hope for your compassion and forgiveness in this matter :) 
</div>