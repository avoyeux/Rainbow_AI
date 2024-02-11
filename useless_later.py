import numpy as np
import pandas as pd
import os
from glob import glob


csv_paths = glob(os.path.join(os.getcwd(), '**', 'statistics.csv'), recursive=True)
print(f'the initial filepaths are {csv_paths}')

for csv_path in csv_paths:
    df = pd.read_csv(csv_path)

    headers = df['Metric']
    items = df['VaryingLSTMKernelSizes_2Inputs']

    new_dict = {}
    for i, header in enumerate(headers):
        new_dict[header] = [items[i]]

    new_df = pd.DataFrame(new_dict)
    new_df.to_csv(os.path.join(os.path.dirname(csv_path), 'statistics_new.csv'), index=False)






