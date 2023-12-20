import pandas as pd
import numpy as np
import os
import random

mask_dict_file = 'class_dict.csv' 
df = pd.read_csv(mask_dict_file)
num_classes = len(df['name'])
print(num_classes)

def random_image():
    test_folder_path = "data/test"
    image_files = [file for file in os.listdir(test_folder_path) if file.endswith(".jpg") or file.endswith(".png")]
    #случайное изображение из списка
    random_image = random.choice(image_files)
    return random_image

image = random_image()
print(image)

data = pd.read_csv('class_dict.csv')
print(data)

