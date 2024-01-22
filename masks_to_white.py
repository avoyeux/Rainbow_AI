import os 
import re 
# from PIL import Image
# import numpy as np
from pathlib import Path 
import shutil

# Init paths
main_path = os.getcwd()
STEREO_path = '/home/avoyeux/old_project/avoyeux/STEREO'
mask_paths = Path(os.path.join(main_path, 'masque_karine_processed')).glob('*.png')
input1_paths = list(Path(os.path.join(STEREO_path, 'int')).glob('*.png'))
input2_paths = list(Path(os.path.join(STEREO_path, 'avg')).glob('*.png'))
# Save paths
save1_path = os.path.join(main_path, 'Inputs_kar')
save2_path = os.path.join(main_path, 'Inputs2_kar')
os.makedirs(save1_path, exist_ok=True)
os.makedirs(save2_path, exist_ok=True)

# Patterns
mask_pattern = re.compile(r'''frame(?P<number>\d{4})\.png''')
input_pattern = re.compile(r'''(?P<number>\d{4})_''')

numbers = sorted([int(mask_pattern.match(os.path.basename(path)).group('number')) for path in mask_paths])


# for path in input1_paths:
#     print(path)
for nb in numbers:
    print(f'number is {nb}')

    for path in input1_paths:
        img_match = input_pattern.match(os.path.basename(path))
        if img_match:
            img_number = int(img_match.group('number'))
            
            if img_number==nb:
                shutil.copy(path, os.path.join(save1_path, os.path.basename(path)))
                break

        else:
            raise ValueError(f'the path {os.path.basename(path)} is wrong')





