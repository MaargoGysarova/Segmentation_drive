import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset , DataLoader
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Загрузка данных из CSV файла
data = pd.read_csv('class_dict.csv')

class MyDataset(Dataset):
    def __init__(self, img_dir: str, mask_dir: str, split: str = 'train', transform=None, is_train=True):
        assert split in ['train', 'val', 'test']
        self.split = split
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.is_train = is_train
        self.num_classes = len(data['name'])

        self.base_transform = transforms.Compose([
    transforms.ToTensor(),  # Преобразование в тензор
    transforms.Resize((256, 256)),  # Измените размер изображений
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Нормализация
])

        self.images_path = sorted([os.path.join(img_dir, item) for item in os.listdir(img_dir)])
        self.masks_path = sorted([os.path.join(mask_dir, item) for item in os.listdir(mask_dir)])

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        img_path = self.images_path[index]
        mask_path = str(self.files[index]).replace(self.split, self.split + '_labels').replace('.png', '_L.png')

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        height, width, _ = image.shape  # Получаем размеры изображения

        # Создаем маску классов на основе class_labels
        mask = np.zeros((height, width, self.num_classes), dtype=np.uint8)
        for i in range(self.num_classes):
            mask[np.where((image[:, :, 0] == data['r'][i]) &
                        (image[:, :, 1] == data['g'][i]) &
                        (image[:, :, 2] == data['b'][i])), i] = 1

        if self.is_train and self.transform is not None:
            augmentations_image = self.transform(image)
            augmentations_mask = self.transform(mask)
            image = augmentations_image
            mask = augmentations_mask

        return image, mask


def get_loaders(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir, batch_size, transform):
    train_dataset = MyDataset(train_img_dir, train_mask_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = MyDataset(val_img_dir, val_mask_dir, transform=transform, is_train= False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader