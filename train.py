# pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cpu/torch_stable.html

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader 
from tqdm import tqdm 
from work_with_dataset import MyDataset , get_loaders
from model import UNET
import pandas as pd
import numpy as np

mask_dict_file = 'class_dict.csv' 
df = pd.read_csv(mask_dict_file)
num_classes = len(df['name'])

# Определите гиперпараметры
learning_rate = 0.001
batch_size = 8
num_epochs = 3

TRAIN_IMG_DIR= "data/train"
TRAIN_MASK_DIR = "data/train_labels"
VAL_IMG_DIR = "data/val"
VAL_MASK_DIR = "data/val_labels"

# IoU, mean IoU, pixel accuracy
# Загрузите данные
transform = transforms.Compose([
    transforms.ToTensor(),  # Преобразование в тензор
    transforms.Resize((256, 256)),  # Измените размер изображений
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Нормализация
])

train_dataset = MyDataset(transform=transform, img_dir=TRAIN_IMG_DIR, mask_dir=TRAIN_MASK_DIR)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_iou(predicted_mask, ground_truth_mask):
    intersection = np.logical_and(predicted_mask, ground_truth_mask)
    union = np.logical_or(predicted_mask, ground_truth_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def calculate_pixel_accuracy(predicted_mask, ground_truth_mask):
    # Количество правильно классифицированных пикселей для всех классов
    correct_pixels = np.sum(predicted_mask == ground_truth_mask)
    # Общее количество пикселей на изображении
    total_pixels = predicted_mask.size
    # Pixel accuracy для всех классов
    pixel_accuracy = correct_pixels / total_pixels
    return pixel_accuracy

# Обучение модели
def train_fn(loader, model, optimizer, criterion, epoch):

    loop = tqdm(loader, position=0)

    model.to(device)

    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    total_iou = 0
    total_pixel_accuracy = 0

    for inputs, masks in train_loader:
        inputs, masks = inputs.to(device), masks.to(device)
        print(inputs.shape)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Вычисление точности
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == masks).sum().item()
        total_samples += masks.numel()

        # Вычисление IoU
        predicted_mask = predicted.cpu().numpy()
        ground_truth_mask = masks.cpu().numpy()
        iou = calculate_iou(predicted_mask, ground_truth_mask)
        total_iou += iou

        # Вычисление pixel accuracy
        pixel_accuracy = calculate_pixel_accuracy(predicted_mask, ground_truth_mask)
        total_pixel_accuracy += pixel_accuracy

        # Обновление прогресс-бара и вывод информации
        loop.set_description(f"Train Epoch [{epoch+1}/{num_epochs}]")
        loop.set_postfix(loss=loss.item(), acc=correct_predictions / total_samples)
        loop.update(1)

    average_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / total_samples
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")


def main():
    model = UNET(in_channels=3, out_channels=num_classes).to(device)
    train_loader, val_loader = get_loaders(TRAIN_IMG_DIR, TRAIN_MASK_DIR,VAL_IMG_DIR, VAL_MASK_DIR, batch_size, transform)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        train_fn(train_loader, model, optimizer, criterion , epoch)

    # Сохранение обученной модели
    torch.save(model.state_dict(), "road_segmentation_model.pth")
    print("Обучение завершено. Модель сохранена как 'road_segmentation_model.pth'.")
        

if __name__ == "__main__":
    main()
