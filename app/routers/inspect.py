import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.spatial.distance import pdist, squareform
import math
from typing import Tuple, Optional, Dict, Any, List, Union


# Калибровочные параметры камеры (пример для Orbbec Astra или аналога)
FX = 1498.922607
FY = 1497.590820
CX = 1273.266724
CY = 728.279663


def get_metrics_coordinates_upper_left(
    u: float, v: float, d: float,
    fx: float = FX, fy: float = FY,
    cx: float = CX, cy: float = CY
) -> Tuple[float, float, float]:
    """
    Преобразует пиксельные координаты (u, v) и глубину d в метрические 3D-координаты (x, y, z).

    Использует стандартную модель камеры: x = (u - cx) * z / fx, y = (v - cy) * z / fy.

    Args:
        u (float): Горизонтальная координата пикселя.
        v (float): Вертикальная координата пикселя.
        d (float): Глубина (z) в миллиметрах.
        fx (float): Фокусное расстояние по оси X (в пикселях).
        fy (float): Фокусное расстояние по оси Y (в пикселях).
        cx (float): Главная точка по оси X.
        cy (float): Главная точка по оси Y.

    Returns:
        tuple[float, float, float]: 3D-координаты (x, y, z) в миллиметрах.
    """
    z = d
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return x, y, z


def get_cucumber_mask(image_path: str) -> Optional[np.ndarray]:
    """
    Создаёт бинарную маску огурца по цвету (зелёный диапазон в HSV).

    Args:
        image_path (str): Путь к изображению.

    Returns:
        np.ndarray or None: Бинарная маска (255 — огурец, 0 — фон), или None при ошибке.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Диапазон зелёного цвета (подходит для огурцов)
    lower_green = np.array([35, 30, 30])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(img_hsv, lower_green, upper_green)
    return mask


def process_mask(
    mask: np.ndarray,
    image_shape: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Tuple[int, int]]:
    """
    Очищает маску от шума и выделяет самый большой контур (огурец).

    Args:
        mask (np.ndarray): Исходная бинарная маска.
        image_shape (tuple): Форма исходного изображения (height, width). Если None — создаётся из маски.

    Returns:
        tuple: Кортеж из:
            - img_rgb: RGB-изображение (заглушка, возвращается как нули).
            - mask: Исходная маска.
            - clean_mask: Очищенная маска (только крупный контур).
            - largest_contour: Контур огурца.
            - image_dimensions: Размеры изображения (height, width).
    """
    # Морфологические операции для удаления шума
    kernel = np.ones((5, 5), np.uint8)
    clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

    # Поиск контуров
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = None

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean_mask, [largest_contour], 0, 255, -1)

    height, width = image_shape if image_shape else mask.shape[:2]
    img_rgb = np.zeros((height, width, 3), dtype=np.uint8)  # Заглушка

    return img_rgb, mask, clean_mask, largest_contour, (height, width)


def create_cucumber_mask(image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Tuple[int, int]]:
    """
    Полный конвейер: загрузка → маска → очистка → контур.

    Args:
        image_path (str): Путь к изображению.

    Returns:
        tuple: Результат `process_mask` (см. выше).
    """
    mask = get_cucumber_mask(image_path)
    if mask is None:
        raise ValueError(f"Could not generate mask for image: {image_path}")
    return process_mask(mask, image_shape=None)


def find_extreme_points(contour: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """
    Находит крайние точки контура: левая/правая, верхняя/нижняя.

    Args:
        contour (np.ndarray): Контур огурца.

    Returns:
        tuple: Две точки — наиболее удалённые друг от друга.
    """
    if contour is None or len(contour) == 0:
        return None, None

    leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
    rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
    topmost = tuple(contour[contour[:, :, 1].argmin()][0])
    bottommost = tuple(contour[contour[:, :, 1].argmax()][0])

    points = np.array([leftmost, rightmost, topmost, bottommost])
    dist_matrix = squareform(pdist(points))
    max_idx = np.unravel_index(dist_matrix.argmax(), dist_matrix.shape)
    p1, p2 = tuple(map(tuple, points[max_idx]))

    return p1, p2


def arc_function(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Параболическая функция для подбора дуги: y = a*(x - b)^2 + c.

    Args:
        x (np.ndarray): Входные координаты.
        a (float): Кривизна.
        b (float): Горизонтальный сдвиг.
        c (float): Вертикальный сдвиг.

    Returns:
        np.ndarray: Выходные значения.
    """
    return a * (x - b)**2 + c


def fit_arc_to_cucumber(
    contour: np.ndarray,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    clean_mask: np.ndarray
) -> Optional[Dict[str, Any]]:
    """
    Подбирает параболическую дугу вдоль контура огурца между двумя крайними точками.

    Учитывает форму маски, чтобы дуга оставалась внутри огурца.

    Args:
        contour (np.ndarray): Контур огурца.
        p1 (tuple): Первая крайняя точка.
        p2 (tuple): Вторая крайняя точка.
        clean_mask (np.ndarray): Очищенная маска.

    Returns:
        dict or None: Результаты подбора (параметры, точки, углы, глубина и т.д.) или None при ошибке.
    """
    if contour is None or p1 is None or p2 is None:
        return None

    contour_points = contour.reshape(-1, 2)
    if p1[0] > p2[0]:
        p1, p2 = p2, p1

    line_vec = np.array(p2) - np.array(p1)
    line_length = np.linalg.norm(line_vec)
    line_unit_vec = line_vec / line_length
    perp_vec = np.array([-line_unit_vec[1], line_unit_vec[0]])

    num_samples = 20
    sample_points = []
    midpoints = []

    for i in range(num_samples):
        t = i / (num_samples - 1)
        point_on_line = np.array(p1) + t * line_vec
        max_dist = line_length * 0.5
        sample_count = 30
        inside_points = []

        for j in range(-sample_count, sample_count + 1):
            dist = j * max_dist / sample_count
            sample_point = point_on_line + dist * perp_vec
            sample_point = (int(round(sample_point[0])), int(round(sample_point[1])))

            if (0 <= sample_point[0] < clean_mask.shape[1] and
                0 <= sample_point[1] < clean_mask.shape[0] and
                clean_mask[sample_point[1], sample_point[0]] > 0):
                inside_points.append(sample_point)

        if inside_points:
            mid_x = np.mean([p[0] for p in inside_points])
            mid_y = np.mean([p[1] for p in inside_points])
            midpoints.append((mid_x, mid_y))
            sample_points.extend(inside_points)

    if len(midpoints) < 5:
        print("Not enough midpoints found, falling back to original method")
        filtered_points = []
        for point in contour_points:
            vec_to_point = point - np.array(p1)
            projection = np.dot(vec_to_point, line_unit_vec)
            if 0 <= projection <= line_length:
                projection_point = np.array(p1) + projection * line_unit_vec
                distance = np.linalg.norm(point - projection_point)
                if distance < line_length * 0.3:
                    filtered_points.append(point)
        if len(filtered_points) < 3:
            return None
        points_for_fitting = np.array(filtered_points)
    else:
        points_for_fitting = np.array(midpoints)

    angle = np.arctan2(line_vec[1], line_vec[0])
    rotation_matrix = np.array([
        [np.cos(-angle), -np.sin(-angle)],
        [np.sin(-angle), np.cos(-angle)]
    ])
    center = (np.array(p1) + np.array(p2)) / 2
    translated_points = points_for_fitting - center
    rotated_points = np.dot(translated_points, rotation_matrix.T)

    x_data = rotated_points[:, 0]
    y_data = rotated_points[:, 1]
    initial_guess = [-0.01, 0, 0]

    try:
        params, _ = optimize.curve_fit(arc_function, x_data, y_data, p0=initial_guess)
        a, b, c = params
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = arc_function(x_fit, a, b, c)
        fit_points = np.column_stack((x_fit, y_fit))
        fit_points_rotated = np.dot(fit_points, rotation_matrix)
        fit_points_final = fit_points_rotated + center

        # Проверка, насколько дуга внутри маски
        points_inside = sum(
            0 <= int(round(p[0])) < clean_mask.shape[1] and
            0 <= int(round(p[1])) < clean_mask.shape[0] and
            clean_mask[int(round(p[1])), int(round(p[0]))] > 0
            for p in fit_points_final
        )
        inside_ratio = points_inside / len(fit_points_final)

        if inside_ratio < 0.8:
            print(f"Initial fit has only {inside_ratio:.2f} points inside mask. Adjusting...")
            best_ratio = inside_ratio
            best_fit_points = fit_points_final
            for factor in np.linspace(0.5, 1.5, 10):
                test_a = a * factor
                y_test = arc_function(x_fit, test_a, b, c)
                test_fit_points = np.column_stack((x_fit, y_test))
                test_fit_rotated = np.dot(test_fit_points, rotation_matrix)
                test_fit_final = test_fit_rotated + center
                test_inside = sum(
                    0 <= int(round(p[0])) < clean_mask.shape[1] and
                    0 <= int(round(p[1])) < clean_mask.shape[0] and
                    clean_mask[int(round(p[1])), int(round(p[0]))] > 0
                    for p in test_fit_final
                )
                test_ratio = test_inside / len(test_fit_final)
                if test_ratio > best_ratio:
                    best_ratio = test_ratio
                    best_fit_points = test_fit_final
            fit_points_final = best_fit_points
            print(f"Adjusted fit has {best_ratio:.2f} points inside mask.")

        # Расчёт глубины дуги
        p1_rotated = np.dot(np.array(p1) - center, rotation_matrix.T)
        p2_rotated = np.dot(np.array(p2) - center, rotation_matrix.T)
        vertex_x = b
        vertex_y = arc_function(vertex_x, a, b, c)
        depth_px = abs(vertex_y)

        # Длина дуги в пикселях
        arc_length_px = sum(
            np.linalg.norm(fit_points_final[i+1] - fit_points_final[i])
            for i in range(len(fit_points_final) - 1)
        )

        # Угол между линией и вертикальной осью (Y)
        topmost_point = p1 if p1[1] < p2[1] else p2
        other_point = p2 if p1[1] < p2[1] else p1
        line_vector = np.array(other_point) - np.array(topmost_point)
        y_axis = np.array([0, 1])
        dot_product = np.dot(line_vector, y_axis)
        line_mag = np.linalg.norm(line_vector)
        y_mag = np.linalg.norm(y_axis)
        angle_rad = np.arccos(dot_product / (line_mag * y_mag))
        line_to_y_angle_deg = np.degrees(angle_rad)

        return {
            'params': (a, b, c),
            'rotated_points': rotated_points,
            'arc_points': fit_points_final,
            'angle': angle,
            'center': center,
            'depth_px': depth_px,
            'p1_rotated': p1_rotated,
            'p2_rotated': p2_rotated,
            'vertex': (vertex_x, vertex_y),
            'sample_points': sample_points,
            'midpoints': midpoints,
            'arc_length_px': arc_length_px,
            'extreme_points': (p1, p2),
            'topmost_point': topmost_point,
            'line_to_y_angle_deg': line_to_y_angle_deg
        }

    except Exception as e:
        print(f"Arc fitting error: {e}")
        return None


def calculate_arc_length(
    arc_result: Optional[Dict[str, Any]],
    extreme_points_distances: Tuple[float, float]
) -> Dict[str, float]:
    """
    Рассчитывает длину дуги, глубину и углы в миллиметрах, используя данные глубины.

    Args:
        arc_result (dict): Результат `fit_arc_to_cucumber`.
        extreme_points_distances (tuple): Глубины (в мм) для двух крайних точек.

    Returns:
        dict: Словарь с измерениями в миллиметрах.
    """
    if arc_result is None:
        return {
            'length_mm': 0.0,
            'depth_mm': 0.0,
            'line_to_y_angle_deg': 0.0,
            'horizontal_angle_deg': 0.0,
            'straight_line_3d_mm': 0.0,
            'straight_line_mm': 0.0,
            'p1_3d': (0, 0, 0),
            'p2_3d': (0, 0, 0)
        }

    p1, p2 = arc_result['extreme_points']
    arc_length_px = arc_result['arc_length_px']
    depth_px = arc_result['depth_px']
    line_to_y_angle_deg = arc_result['line_to_y_angle_deg']

    d1, d2 = extreme_points_distances
    p1_3d = get_metrics_coordinates_upper_left(p1[0], p1[1], d1)
    p2_3d = get_metrics_coordinates_upper_left(p2[0], p2[1], d2)
    p1_3d_arr = np.array(p1_3d)
    p2_3d_arr = np.array(p2_3d)
    straight_line_3d_mm = np.linalg.norm(p2_3d_arr - p1_3d_arr)

    straight_line_px = np.linalg.norm(np.array(p2) - np.array(p1))
    scale_factor = straight_line_3d_mm / straight_line_px if straight_line_px > 0 else 1.0

    arc_length_mm = arc_length_px * scale_factor
    depth_mm = depth_px * scale_factor

    line_vector_3d = p2_3d_arr - p1_3d_arr
    line_vector_xz = np.array([line_vector_3d[0], 0, line_vector_3d[2]])
    line_mag_3d = np.linalg.norm(line_vector_3d)
    xz_mag = np.linalg.norm(line_vector_xz)

    if xz_mag > 0:
        cos_angle = np.dot(line_vector_3d, line_vector_xz) / (line_mag_3d * xz_mag)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        horizontal_angle_rad = np.arccos(cos_angle)
        if line_vector_3d[1] < 0:
            horizontal_angle_rad = -horizontal_angle_rad
        horizontal_angle_deg = np.degrees(horizontal_angle_rad)
    else:
        horizontal_angle_deg = 90.0 if line_vector_3d[1] > 0 else -90.0

    return {
        'length_mm': float(arc_length_mm),
        'depth_mm': float(depth_mm),
        'line_to_y_angle_deg': float(line_to_y_angle_deg),
        'horizontal_angle_deg': float(horizontal_angle_deg),
        'straight_line_3d_mm': float(straight_line_3d_mm),
        'straight_line_mm': float(straight_line_px * scale_factor),
        'p1_3d': p1_3d,
        'p2_3d': p2_3d
    }


def analyze_cucumber(
    image_path: str,
    extreme_points_distances: Tuple[float, float]
) -> Optional[Dict[str, float]]:
    """
    Полный анализ огурца: маска → контур → дуга → измерения.

    Args:
        image_path (str): Путь к изображению.
        extreme_points_distances (tuple): Глубины для крайних точек (в мм).

    Returns:
        dict or None: Измерения длины, глубины, углов.
    """
    try:
        _, _, clean_mask, contour, _ = create_cucumber_mask(image_path)
        if contour is None:
            print("Failed to detect cucumber contour.")
            return None

        p1, p2 = find_extreme_points(contour)
        arc_result = fit_arc_to_cucumber(contour, p1, p2, clean_mask)
        measurements = calculate_arc_length(arc_result, extreme_points_distances)

        print(f"Cucumber length: {measurements['length_mm']:.1f} mm")
        print(f"Cucumber depth: {measurements['depth_mm']:.1f} mm")
        print(f"Angle to Y-axis: {measurements['line_to_y_angle_deg']:.1f}°")
        print(f"Angle to horizontal plane: {measurements['horizontal_angle_deg']:.1f}°")

        return measurements

    except Exception as e:
        print(f"Error during analysis: {e}")
        return None


# === Пример использования ===
if __name__ == "__main__":
    image_path = "path/to/cucumber_image.jpg"  # Замените на реальный путь
    extreme_points_distances = (500, 520)  # Пример: глубина в мм

    measurements = analyze_cucumber(image_path, extreme_points_distances)
