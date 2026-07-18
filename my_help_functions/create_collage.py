import json
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO


def create_collage(seed=42, save=True):

    def set_random_seed(seed):
        random.seed(seed)
        np.random.seed(seed)

    def load_coco_data(coco_ann_path, coco_img_dir):
        coco = COCO(coco_ann_path)
        img_ids = coco.getImgIds()
        return coco, img_ids, coco_img_dir

    def get_objects_from_different_classes(coco, img_ids, num_objects=50):
        selected_objects = []
        used_categories = set()

        random.shuffle(img_ids)
        for img_id in img_ids:
            img_info = coco.loadImgs(img_id)[0]
            img_area = img_info['width'] * img_info['height']
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)

            for ann in anns:
                cat_id = ann['category_id']
                bbox = ann['bbox']
                bbox_area = bbox[2] * bbox[3]
                area_ratio = bbox_area / img_area

                if cat_id not in used_categories and 0.1 <= area_ratio <= 0.2:
                    selected_objects.append((img_id, ann))
                    used_categories.add(cat_id)
                    if len(selected_objects) == num_objects:
                        return selected_objects
        return selected_objects

    def extract_bbox_objects(coco, img_dir, objects):
        extracted_objects = []
        for img_id, ann in objects:
            img_info = coco.loadImgs(img_id)[0]
            img_path = f"{img_dir}/{img_info['file_name']}"
            img = cv2.imread(img_path)
            x, y, w, h = map(int, ann['bbox'])
            cropped_obj = img[y:y+h, x:x+w]
            extracted_objects.append((cropped_obj, ann['category_id'], ann['bbox']))
        return extracted_objects

    def get_background_patch(img, occupied_regions):
        h, w, _ = img.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        for x, y, x2, y2 in occupied_regions:
            mask[y:y2, x:x2] = 1

        patch_size = min(h, w) // 5
        corners = [(0, 0), (0, w - patch_size), (h - patch_size, 0), (h - patch_size, w - patch_size)]
        random.shuffle(corners)

        for y, x in corners:
            if np.all(mask[y:y+patch_size, x:x+patch_size] == 0):
                return img[y:y+patch_size, x:x+patch_size], [x, y, patch_size, patch_size]
        return None, None

    def create_collage(bg_img, objects, occupied_regions):
        h, w, _ = bg_img.shape
        collage = bg_img.copy()
        positions = []
        placed_regions = occupied_regions.copy()
        bg_mask = np.zeros((h, w), dtype=np.uint8)

        added_objects = 0
        max_attempts_per_object = 50
        random.shuffle(objects)

        while added_objects < 3 and objects:
            obj, class_id, bbox = objects.pop(0)
            oh, ow, _ = obj.shape
            if oh > h or ow > w:
                continue  # Пропускаем слишком большие объекты

            for _ in range(max_attempts_per_object):
                x, y = random.randint(0, w-ow), random.randint(0, h-oh)
                region = (x, y, x+ow, y+oh)

                if not any(overlaps(region, placed) for placed in placed_regions) and np.all(bg_mask[y:y+oh, x:x+ow] == 0):
                    placed_regions.append(region)
                    bg_mask[y:y+oh, x:x+ow] = 1
                    collage[y:y+oh, x:x+ow] = obj
                    positions.append({'bbox': [x, y, ow, oh], 'category_id': class_id})
                    added_objects += 1
                    break

        return collage, positions, placed_regions

    def overlaps(region1, region2):
        x1_min, y1_min, x1_max, y1_max = region1
        x2_min, y2_min, x2_max, y2_max = region2
        return not (x1_max <= x2_min or x1_min >= x2_max or y1_max <= y2_min or y1_min >= y2_max)

    set_random_seed(seed)

    # Загрузка COCO
    coco_ann_path = './datasets/coco/' + 'annotations/instances_val2017.json'
    coco_img_dir = './datasets/coco/val2017'
    coco, img_ids, img_dir = load_coco_data(coco_ann_path, coco_img_dir)

    # Фон
    bg_img_info = coco.loadImgs(785)[0]
    bg_img_path = f"{img_dir}/{bg_img_info['file_name']}"
    bg_img = cv2.imread(bg_img_path)

    # Получение объектов
    objects = get_objects_from_different_classes(coco, img_ids)
    extracted_objects = extract_bbox_objects(coco, img_dir, objects)

    # Определение уже занятых областей на фоновой картинке
    occupied_regions = [
        (int(ann['bbox'][0]), int(ann['bbox'][1]), int(ann['bbox'][0] + ann['bbox'][2]), int(ann['bbox'][1] + ann['bbox'][3]))
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=785))
    ]

    # Создание коллажа
    collage, new_annotations, final_occupied_regions = create_collage(bg_img, extracted_objects, occupied_regions)

    # Добавление объектов из фонового изображения в аннотации
    new_annotations.extend([{'bbox': ann['bbox'], 'category_id': ann['category_id']} for ann in coco.loadAnns(coco.getAnnIds(imgIds=785))])

    # Получение фонового патча
    bg_patch, bg_patch_bbox = get_background_patch(collage, final_occupied_regions)
    if bg_patch is not None:
        background_annotation = {'bbox': bg_patch_bbox, 'category_id': 'background'}
        new_annotations.append(background_annotation)

    # Вывод названий классов
    class_names = [coco.loadCats(ann['category_id'])[0]['name'] if ann['category_id'] != 'background' else 'background' for ann in new_annotations]
    print("Классы в new_annotations:", class_names)

    # Сохранение результата
    if save:
        cv2.imwrite(f"collage/collage_{seed}.jpg", collage)
        with open(f"collage/annotations_{seed}.json", "w") as f:
            json.dump(new_annotations, f, indent=4)

    # Отображение результата
    plt.imshow(cv2.cvtColor(collage, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


def create_collage_gray_back(seed=42, save=True, required_objs=5):

    def set_random_seed(seed):
        random.seed(seed)
        np.random.seed(seed)

    def load_coco_data(coco_ann_path, coco_img_dir):
        coco = COCO(coco_ann_path)
        img_ids = coco.getImgIds()
        return coco, img_ids, coco_img_dir

    def get_objects_from_different_classes(coco, img_ids, num_objects=50):
        selected_objects = []
        used_categories = set()

        random.shuffle(img_ids)
        for img_id in img_ids:
            img_info = coco.loadImgs(img_id)[0]
            img_area = img_info['width'] * img_info['height']
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)

            for ann in anns:
                cat_id = ann['category_id']
                bbox = ann['bbox']
                bbox_area = bbox[2] * bbox[3]
                area_ratio = bbox_area / img_area

                if cat_id not in used_categories and 0.1 <= area_ratio <= 0.2:
                    selected_objects.append((img_id, ann))
                    used_categories.add(cat_id)
                    if len(selected_objects) == num_objects:
                        return selected_objects
        return selected_objects

    def extract_bbox_objects(coco, img_dir, objects):
        extracted_objects = []
        for img_id, ann in objects:
            img_info = coco.loadImgs(img_id)[0]
            img_path = f"{img_dir}/{img_info['file_name']}"
            img = cv2.imread(img_path)
            x, y, w, h = map(int, ann['bbox'])
            cropped_obj = img[y:y+h, x:x+w]
            extracted_objects.append((cropped_obj, ann['category_id'], ann['bbox']))
        return extracted_objects

    def create_collage(bg_img, objects, required_objs=5):
        h, w, _ = bg_img.shape
        collage = bg_img.copy()
        positions = []
        placed_regions = []
        bg_mask = np.zeros((h, w), dtype=np.uint8)

        added_objects = 0
        max_attempts_per_object = 50
        random.shuffle(objects)

        while added_objects < required_objs and objects:
            obj, class_id, bbox = objects.pop(0)
            oh, ow, _ = obj.shape
            if oh > h or ow > w:
                continue  # Пропускаем слишком большие объекты

            for _ in range(max_attempts_per_object):
                x, y = random.randint(0, w-ow), random.randint(0, h-oh)
                region = (x, y, x+ow, y+oh)

                if not any(overlaps(region, placed) for placed in placed_regions) and np.all(bg_mask[y:y+oh, x:x+ow] == 0):
                    placed_regions.append(region)
                    bg_mask[y:y+oh, x:x+ow] = 1
                    collage[y:y+oh, x:x+ow] = obj
                    positions.append({'bbox': [x, y, ow, oh], 'category_id': class_id})
                    added_objects += 1
                    break

        return collage, positions, placed_regions

    def overlaps(region1, region2):
        x1_min, y1_min, x1_max, y1_max = region1
        x2_min, y2_min, x2_max, y2_max = region2
        return not (x1_max <= x2_min or x1_min >= x2_max or y1_max <= y2_min or y1_min >= y2_max)

    def get_background_patch(img, occupied_regions):
        h, w, _ = img.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        for x, y, x2, y2 in occupied_regions:
            mask[y:y2, x:x2] = 1

        patch_size = min(h, w) // 5
        corners = [(0, 0), (0, w - patch_size), (h - patch_size, 0), (h - patch_size, w - patch_size)]
        random.shuffle(corners)

        for y, x in corners:
            if np.all(mask[y:y+patch_size, x:x+patch_size] == 0):
                return img[y:y+patch_size, x:x+patch_size], [x, y, patch_size, patch_size]
        return None, None

    set_random_seed(seed)

    # Параметры
    img_size = 640
    gray_color = 128

    # Загрузка COCO
    coco_ann_path = './datasets/coco/' + 'annotations/instances_val2017.json'
    coco_img_dir = './datasets/coco/val2017'
    coco, img_ids, img_dir = load_coco_data(coco_ann_path, coco_img_dir)

    # Создание серого фона
    bg_img = np.full((img_size, img_size, 3), gray_color, dtype=np.uint8)

    # Получение объектов
    objects = get_objects_from_different_classes(coco, img_ids)
    extracted_objects = extract_bbox_objects(coco, img_dir, objects)

    # Создание коллажа
    collage, new_annotations, final_occupied_regions = create_collage(bg_img, extracted_objects, required_objs)

    # Получение фонового патча
    bg_patch, bg_patch_bbox = get_background_patch(collage, final_occupied_regions)
    if bg_patch is not None:
        background_annotation = {'bbox': bg_patch_bbox, 'category_id': 'background'}
        new_annotations.append(background_annotation)

    # Вывод названий классов
    class_names = [coco.loadCats(ann['category_id'])[0]['name'] if ann['category_id'] != 'background' else 'background' for ann in new_annotations]
    print("Классы в new_annotations:", class_names)

    # Сохранение результата
    if save:
        cv2.imwrite(f"collage/gray_collage_{seed}.jpg", collage)
        with open(f"collage/gray_annotations_{seed}.json", "w") as f:
            json.dump(new_annotations, f, indent=4)

    # Отображение результата
    plt.imshow(cv2.cvtColor(collage, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
