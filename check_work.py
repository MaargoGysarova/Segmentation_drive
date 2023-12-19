import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import random

def random_image():
    test_folder_path = "data/test"
    image_files = [file for file in os.listdir(test_folder_path) if file.endswith(".jpg") or file.endswith(".png")]
    #случайное изображение из списка
    random_image = random.choice(image_files)
    return random_image

# Загрузите обученную модель
from model import UNET  # Замените на ваш импорт модели
model = UNET()  # Создайте экземпляр модели
model.load_state_dict(torch.load("road_segmentation_model.pth"))  # Загрузите веса модели
model.eval()  # Установите модель в режим оценки (не обучения)

# Определите преобразования для входных изображений
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Загрузите изображение для сегментации
image_path = os.path.join("data/test", random_image())
image = Image.open(image_path)
image = transform(image)  # Примените преобразования

# Прогоните изображение через модель
with torch.no_grad():
    image = image.unsqueeze(0)  # пакет = 0 так как мы работаем с одним изображением
    output = model(image)

# Получим сегментированную маску из выхода модели
segmentation_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()


# Визуализируем сегментированную маску и исходное изображение
plt.figure(figsize=(12, 8))
plt.imshow(segmentation_mask, cmap='gray')
plt.axis('off')
plt.title("Сегментированная маска")
plt.show()

plt.figure(figsize=(12, 8))
plt.imshow(image.squeeze().permute(1, 2, 0))
plt.axis('off')
plt.title("Исходное изображение")
plt.show()

