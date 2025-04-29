import cv2
import numpy as np
from pathlib import Path
import os
from pycocotools.coco import COCO
import json


def get_positions_of_classes_on_flattened_image(name, size):
    im = cv2.imread('./datasets/coco/val2017/' + name)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype(float) / 255.

    ann_file = './datasets/coco/annotations/instances_val2017.json'
    coco = COCO(ann_file)

    # Изменяем размер до size
    image_resized = cv2.resize(im, (size, size))

    img_info = next(img for img in coco.dataset["images"] if img["file_name"] == name)
    img_id = img_info["id"]
    # Загружаем все bbox-ы для этого изображения
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    # Создаём пустую маску (0 - фон, 1 - объекты)
    mask = np.zeros((size, size), dtype=np.uint8)

    # Наносим bbox на маску, каждому классу даём уникальный индекс
    class_to_id = {}  # Словарь: class_id -> уникальный индекс
    next_class_id = 1  # Начинаем индексацию классов с 1

    for ann in anns:
        coco_class_id = ann["category_id"]  # ID класса в COCO
        if coco_class_id not in class_to_id:
            class_to_id[coco_class_id] = next_class_id
            next_class_id += 1

        mask_value = class_to_id[coco_class_id]  # Уникальный ID класса на маске
        x, y, w, h = map(int, ann["bbox"])

        # Масштабируем bbox под size
        x = int(x * size / img_info["width"])
        y = int(y * size / img_info["height"])
        w = int(w * size / img_info["width"])
        h = int(h * size / img_info["height"])

        # Заполняем область bbox значением класса
        mask[y:y+h, x:x+w] = mask_value


    mask = mask.flatten()
    positions = {}
    for class_id in np.unique(mask):  # Перебираем уникальные значения (0, 1, 2, 3...)
        if class_id == 0:
            continue  # Пропускаем фон
        positions[class_id] = np.argwhere(mask == class_id).flatten()

    class_names = {i + 1: coco.loadCats(ann['category_id'])[0]['name'] if ann['category_id'] != 'background' else 'background' for (i, ann) in enumerate(anns)}

    return positions, class_names


def get_positions_of_classes_on_flattened_image_for_collage(idx, size, pref="", postf=""):
    im = cv2.imread(f"./collage/{pref}collage_{postf}{idx}.jpg")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype(float) / 255.

    ann_file = f"./collage/{pref}annotations_{postf}{idx}.json"  # Используем аннотации коллажа
    with open(ann_file, "r") as f:
        annotations = json.load(f)

    # Изменяем размер до size
    image_resized = cv2.resize(im, (size, size))

    # Создаём пустую маску (0 - фон, 1 - объекты)
    mask = np.zeros((size, size), dtype=np.uint8)

    # Наносим bbox на маску, каждому классу даём уникальный индекс
    class_to_id = {}  # Словарь: category_id -> уникальный индекс
    next_class_id = 1  # Начинаем индексацию классов с 1

    for ann in annotations:
        coco_class_id = ann["category_id"]  # ID класса в аннотации
        if coco_class_id not in class_to_id:
            class_to_id[coco_class_id] = next_class_id
            next_class_id += 1
        
        mask_value = class_to_id[coco_class_id]  # Уникальный ID класса на маске
        x, y, w, h = map(int, ann["bbox"])
        
        # Масштабируем bbox под size
        x = int(x * size / im.shape[1])
        y = int(y * size / im.shape[0])
        w = int(w * size / im.shape[1])
        h = int(h * size / im.shape[0])
        
        # Заполняем область bbox значением класса
        mask[y:y+h, x:x+w] = mask_value

    mask = mask.flatten()
    positions = {}
    for class_id in np.unique(mask):  # Перебираем уникальные значения (0, 1, 2, 3...)
        if class_id == 0:
            continue  # Пропускаем фон

        positions[class_id] = np.argwhere(mask == class_id).flatten()

    ann_file = './datasets/coco/annotations/instances_val2017.json'
    coco = COCO(ann_file)

    class_names = {i + 1: coco.loadCats(ann['category_id'])[0]['name'] if ann['category_id'] != 'background' else 'background' for (i, ann) in enumerate(annotations)}

    return positions, class_names
