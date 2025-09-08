import os
import io
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image as PILImage
from IPython.display import Image, display
import matplotlib.pyplot as plt
from ultralytics import YOLO
from scipy.spatial.distance import cdist
from tqdm import tqdm
from sklearn.linear_model import LinearRegression


def get_all_images(image_dir: str) -> List[Path]:
    """
    Находит все изображения в указанной директории с поддерживаемыми расширениями.

    Args:
        image_dir (str): Путь к директории.

    Returns:
        list[Path]: Список путей к изображениям.
    """
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend(list(Path(image_dir).glob(f'*{ext}')))
        image_files.extend(list(Path(image_dir).glob(f'*{ext.upper()}')))
    return sorted(image_files)


def display_image_with_size(
    image_path: Optional[str] = None,
    image_array: Optional[np.ndarray] = None,
    width: int = 800,
    height: int = 600
) -> None:
    """
    Отображает изображение в Jupyter Notebook с заданными размерами.

    Args:
        image_path (str, optional): Путь к файлу изображения.
        image_array (np.ndarray, optional): Изображение в виде массива.
        width (int): Ширина отображения.
        height (int): Высота отображения.
    """
    if image_path is not None:
        img = PILImage.open(image_path)
    elif image_array is not None:
        if image_array.ndim == 3 and image_array.shape[2] == 3:
            img = PILImage.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
        else:
            img = PILImage.fromarray(image_array)
    else:
        raise ValueError("Either image_path or image_array must be provided")

    img = img.resize((width, height), PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    display(Image(data=buf.getvalue(), width=width, height=height))


def get_mask_multitype(
    mask_obj: Any,
    img: np.ndarray,
    cls_id: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Преобразует маску из YOLO в бинарную маску, цветную маску и наложенное изображение.

    Args:
        mask_obj: Объект маски из YOLO.
        img (np.ndarray): Оригинальное изображение (BGR).
        cls_id (int): ID класса (0 — огурец, 1 — стебель).

    Returns:
        tuple: (binary_mask, colored_mask, overlay_image)
    """
    class_colors = {
        0: (0, 255, 0),  # Зелёный — огурец
        1: (255, 0, 0)   # Красный — стебель
    }

    assert len(mask_obj.data) == 1, "Compound mask found"
    assert mask_obj.orig_shape == img.shape[:2], "Mask shape mismatch"

    mask_tensor = mask_obj.data[0].cpu().numpy()
    img_h, img_w = img.shape[:2]

    # Изменение размера маски, если необходимо
    if mask_tensor.shape != (img_h, img_w):
        mask = cv2.resize(mask_tensor, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    else:
        mask = mask_tensor

    color = class_colors.get(cls_id, (255, 255, 255))
    colored_mask = np.zeros_like(img)
    colored_mask[mask > 0] = color

    overlay = img.copy()
    alpha = 0.5
    overlay[mask > 0] = overlay[mask > 0] * (1 - alpha) + colored_mask[mask > 0] * alpha

    return mask, colored_mask, overlay


def get_stem_center_line(
    stem_mask: np.ndarray,
    image_shape: Optional[Tuple[int, int]] = None
) -> Tuple[Any, Tuple[float, float]]:
    """
    Вычисляет центральную линию стебля с помощью линейной регрессии.

    Args:
        stem_mask (np.ndarray): Бинарная маска стебля.
        image_shape (tuple, optional): Размер изображения.

    Returns:
        tuple: (функция линии, (наклон, пересечение))
    """
    if image_shape is None:
        image_shape = stem_mask.shape

    contours, _ = cv2.findContours(stem_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        center_x = stem_mask.shape[1] // 2
        return lambda x: np.full_like(x, center_x), (0, center_x)

    contour = max(contours, key=cv2.contourArea)
    points = contour.reshape(-1, 2)

    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m01"])
    else:
        cx, cy = np.mean(points, axis=0).astype(int)

    if len(points) > 1:
        X = points[:, 1].reshape(-1, 1)  # y
        y = points[:, 0]                  # x
        model = LinearRegression().fit(X, y)
        slope = model.coef_[0]
        intercept = model.intercept_

        if slope == 0:
            m = float('inf')
            b = cy
        else:
            m = 1 / slope
            b = -intercept / slope
    else:
        m, b = 0, cy

    def line_func(x):
        if m == float('inf'):
            return np.full_like(x, b)
        return m * x + b

    return line_func, (m, b)


def determine_cucumber_side(
    cucumber_mask: np.ndarray,
    relationship_details: Dict[str, Any]
) -> str:
    """
    Определяет, с какой стороны от стебля находится огурец (лево/право).

    Args:
        cucumber_mask (np.ndarray): Маска огурца.
        relationship_details (dict): Детали взаимоотношения (точки, тип расстояния).

    Returns:
        str: 'left' или 'right'.
    """
    cucumber_point = np.array(relationship_details['cucumber_point'])
    stem_point = np.array(relationship_details['stem_point'])
    stem_mask = relationship_details.get('stem_mask')

    # Получение ориентации стебля
    if stem_mask is not None:
        _, (m, b) = get_stem_center_line(stem_mask)
    else:
        m, b = float('inf'), stem_point[0]

    if m == float('inf'):
        return 'right' if cucumber_point[0] > stem_point[0] else 'left'

    perp_vector = np.array([-1, m])
    if m < 0:
        perp_vector = -perp_vector

    direction_vector = cucumber_point - stem_point
    dot_product = np.dot(direction_vector, perp_vector)

    return 'right' if dot_product > 0 else 'left'


def find_closest_stem(
    cucumber_mask: np.ndarray,
    cucumber_pos: int,
    stem_masks: List[np.ndarray],
    img: np.ndarray,
    max_distance_factor: float = 2.0,
    max_line_distance_factor: float = 3.0
) -> Optional[Dict[str, Any]]:
    """
    Находит ближайший стебель к огурцу по маске.

    Args:
        cucumber_mask (np.ndarray): Маска огурца.
        cucumber_pos (int): Позиция огурца в списке.
        stem_masks (list): Список масок стеблей.
        img (np.ndarray): Исходное изображение.
        max_distance_factor (float): Макс. расстояние как доля диагонали огурца.
        max_line_distance_factor (float): Макс. расстояние до линии стебля.

    Returns:
        dict or None: Информация о ближайшем стебле или None.
    """
    img_h, img_w = img.shape[:2]
    cucumber_contours, _ = cv2.findContours(cucumber_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cucumber_contours:
        return None

    cucumber_contour = max(cucumber_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cucumber_contour)
    cucumber_diagonal = np.sqrt(w**2 + h**2)

    M = cv2.moments(cucumber_contour)
    cucumber_centroid = np.array([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]) if M["m00"] != 0 else np.array([x + w//2, y + h//2])

    max_distance = cucumber_diagonal * max_distance_factor
    max_line_distance = cucumber_diagonal * max_line_distance_factor
    cucumber_points = cucumber_contour.reshape(-1, 2)

    closest_stem = None
    min_distance = float('inf')
    relationship_details = {}

    for i, stem_mask in enumerate(stem_masks):
        stem_contours, _ = cv2.findContours(stem_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not stem_contours:
            continue

        stem_contour = max(stem_contours, key=cv2.contourArea)
        M = cv2.moments(stem_contour)
        stem_centroid = np.array([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]) if M["m00"] != 0 else np.array([x + w//2, y + h//2])

        # Проверка пересечения масок
        if np.logical_and(cucumber_mask, stem_mask).any():
            dist = np.linalg.norm(cucumber_centroid - stem_centroid)
            if dist < min_distance:
                min_distance = dist
                closest_stem = i
                relationship_details = {
                    'distance': dist,
                    'cucumber_point': cucumber_centroid,
                    'stem_point': stem_centroid,
                    'distance_type': 'centroid',
                    'stem_mask': stem_mask
                }
        else:
            stem_points = stem_contour.reshape(-1, 2)
            distances = cdist(cucumber_points, stem_points)
            min_dist = np.min(distances)
            if min_dist <= max_distance and min_dist < min_distance:
                min_distance = min_dist
                closest_stem = i
                cuc_pt, stem_pt = np.unravel_index(np.argmin(distances), distances.shape)
                relationship_details = {
                    'distance': min_dist,
                    'cucumber_point': cucumber_points[cuc_pt],
                    'stem_point': stem_points[stem_pt],
                    'distance_type': 'mask'
                }
            else:
                # Проверка по линии стебля
                line_func, line_params = get_stem_center_line(stem_mask, (img_h, img_w))
                m, b = line_params
                if m == float('inf'):
                    line_distances = np.abs(cucumber_points[:, 0] - b)
                else:
                    a, b_line, c = m, -1, b
                    line_distances = np.abs(a * cucumber_points[:, 0] + b_line * cucumber_points[:, 1] + c) / np.sqrt(a**2 + b_line**2)
                min_line_dist = np.min(line_distances)
                if min_line_dist <= max_line_distance and min_line_dist < min_distance:
                    min_distance = min_line_dist
                    closest_stem = i
                    closest_pt = np.argmin(line_distances)
                    cuc_pt = cucumber_points[closest_pt]
                    if m == float('inf'):
                        stem_pt = np.array([b, cuc_pt[1]])
                    else:
                        x0, y0 = cuc_pt
                        x1 = (x0 + m * y0 - m * b) / (1 + m**2)
                        y1 = m * x1 + b
                        stem_pt = np.array([x1, y1])
                    relationship_details = {
                        'distance': min_line_dist,
                        'cucumber_point': cuc_pt,
                        'stem_point': stem_pt,
                        'distance_type': 'line',
                        'stem_line': line_params
                    }

    if closest_stem is None:
        return None

    return {
        'cucumber': cucumber_pos,
        'stem': closest_stem,
        'side': determine_cucumber_side(cucumber_mask, relationship_details),
        'distance': min_distance,
        'relationship': relationship_details
    }


def visualize_relationships(
    image: np.ndarray,
    cucumber_masks: List[np.ndarray],
    stem_masks: List[np.ndarray],
    relationships: List[Optional[Dict]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Визуализирует связи между огурцами и стеблями.

    Args:
        image (np.ndarray): Оригинальное изображение.
        cucumber_masks (list): Маски огурцов.
        stem_masks (list): Маски стеблей.
        relationships (list): Список связей.

    Returns:
        tuple: (изображение с масками, изображение с линиями).
    """
    vis_image = image.copy()
    lines_image = image.copy()

    # Наложение масок
    for mask in cucumber_masks:
        vis_image[mask > 0] = vis_image[mask > 0] * 0.7 + np.array([0, 255, 0]) * 0.3
    for mask in stem_masks:
        vis_image[mask > 0] = vis_image[mask > 0] * 0.7 + np.array([255, 0, 0]) * 0.3

    # Центральные линии стеблей
    for mask in stem_masks:
        line_func, (m, b) = get_stem_center_line(mask, image.shape[:2])
        h, w = image.shape[:2]
        if m == float('inf'):
            x = int(b)
            cv2.line(lines_image, (x, 0), (x, h-1), (255, 255, 0), 2)
        else:
            y1 = int(m * 0 + b)
            y2 = int(m * (w-1) + b)
            if 0 <= y1 < h and 0 <= y2 < h:
                cv2.line(lines_image, (0, y1), (w-1, y2), (255, 255, 0), 2)

    # Линии и подписи
    for rel in relationships:
        if rel:
            cuc_pt = rel['relationship']['cucumber_point'].astype(int)
            stem_pt = rel['relationship']['stem_point'].astype(int)
            cv2.line(lines_image, tuple(cuc_pt), tuple(stem_pt), (0, 255, 255), 2)
            mid_pt = ((cuc_pt[0] + stem_pt[0]) // 2, (cuc_pt[1] + stem_pt[1]) // 2)
            text = f"C:{rel['cucumber']} S:{rel['stem']} {rel['side']}"
            cv2.putText(lines_image, text, mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return vis_image, lines_image


def resize_image(image_array: np.ndarray, new_width: int = None, new_height: int = None) -> np.ndarray:
    """Изменяет размер изображения."""
    if new_width and new_height:
        return cv2.resize(image_array, (new_width, new_height))
    return image_array


def load_image(file_path: str) -> np.ndarray:
    """Загружает изображение с диска."""
    return cv2.imread(file_path)


def load_resize_image(file_path: str, new_width: int = None, new_height: int = None) -> np.ndarray:
    """Загружает и изменяет размер изображения."""
    return resize_image(load_image(file_path), new_width, new_height)


def display_images_with_titles(
    images: List[np.ndarray],
    titles: List[str],
    output_path: Optional[str] = None
) -> plt.Figure:
    """Отображает несколько изображений с заголовками."""
    fig, axs = plt.subplots(len(images), 1, figsize=(10, 12))
    if len(images) == 1:
        axs = [axs]
    for ax, img, title in zip(axs, images, titles):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=14)
        ax.axis('on')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    return fig


def get_compare_data_images(result: Any) -> Optional[Tuple]:
    """
    Извлекает данные для сравнения моделей.

    Returns:
        tuple: (lines_image, vis_image, orig_img, relationships, overlaid_masks, raw_masks)
    """
    try:
        cucumber_ixs = [i for i, cls in enumerate(result.boxes.cls) if int(cls) == 0]
        stem_ixs = [i for i, cls in enumerate(result.boxes.cls) if int(cls) == 1]
        if not result.masks:
            return None

        orig_img = cv2.cvtColor(result.orig_img, cv2.COLOR_RGB2BGR)
        cucumber_mask_objs = [result.masks[i] for i in cucumber_ixs]
        stem_mask_objs = [result.masks[i] for i in stem_ixs]

        cucumber_masks = [get_mask_multitype(x, orig_img, 0)[0] for x in cucumber_mask_objs]
        stem_masks = [get_mask_multitype(x, orig_img, 1)[0] for x in stem_mask_objs]
        relationships = [find_closest_stem(c_mask, i, stem_masks, orig_img) for i, c_mask in enumerate(cucumber_masks)]

        cucumber_overlaid = [get_mask_multitype(x, orig_img, 0)[2] for x in cucumber_mask_objs]
        stem_overlaid = [get_mask_multitype(x, orig_img, 1)[2] for x in stem_mask_objs]

        vis_image, lines_image = visualize_relationships(orig_img, cucumber_masks, stem_masks, relationships)

        return (
            lines_image,
            vis_image,
            orig_img,
            relationships,
            [cucumber_overlaid, stem_overlaid],
            [cucumber_masks, stem_masks]
        )
    except Exception as e:
        print(f"Error in get_compare_data_images: {e}")
        return None


def get_examples_comparison(
    image_paths: List[Path],
    models: List[YOLO],
    titles: List[str],
    folder: str,
    sample_sz: int
) -> List[List[Optional[Tuple]]]:
    """
    Генерирует сравнительные изображения для нескольких моделей.

    Args:
        image_paths (list): Список путей к изображениям.
        models (list): Список моделей YOLO.
        titles (list): Заголовки для моделей.
        folder (str): Папка для сохранения.
        sample_sz (int): Количество изображений.

    Returns:
        list: Результаты сравнения.
    """
    os.makedirs(folder, exist_ok=True)
    all_compares = []

    for i in tqdm(range(min(sample_sz, len(image_paths)))):
        img_path = str(image_paths[i])
        image_orig = load_image(img_path)
        results = [
            m.predict(image_orig, conf=0.3, imgsz=1280, save=False, show=False)[0]
            for m in models
        ]
        compares = [get_compare_data_images(r) for r in results]
        all_compares.append(compares)

        lines = [resize_image(comp[0], 1920, 1080) if comp else image_orig for comp in compares]
        output_path = os.path.join(folder, f'example_{i}.jpg')
        display_images_with_titles(lines, titles, output_path=output_path)

    return all_compares
