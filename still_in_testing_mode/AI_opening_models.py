from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt
from AI_data_gen import DataGen

main_path = '/home/avoyeux/AI_tests'


model_list = ['UNetModel', 'UNetDeepResidual']

kernel_list = ['3_3', '5_5', '7_7', '9_9']

epochs_list = [10, 50, 100, 200, 300]

test = DataGen()
_1, _2, test_input, test_output = test.Give_data()


def Prediction(path):
    
    model = keras.models.load_model(os.path.join(path, 'model.h5'))
    result = model.predict(test_input)

    return result 

print()
print(os.path.join(main_path, model_list[0], f'kernel{kernel_list[0]}',
                                 f'Epochs{epochs_list[1]}'))
print()
result = Prediction(os.path.join(main_path, model_list[0], f'kernel{kernel_list[0]}',
                                 f'Epochs{epochs_list[1]}'))


plt.figure()
plt.imshow(result)
plt.colorbar()
plt.savefig(os.path.join(os.getcwd(), 'test.png'), dpi=300)
plt.close()