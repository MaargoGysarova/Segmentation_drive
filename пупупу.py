import pandas as pd

mask_dict_file = 'class_dict.csv' 
df = pd.read_csv(mask_dict_file)
num_classes = len(df['name'])
print(num_classes)