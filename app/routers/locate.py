import asyncio
import time
import os
import pickle
import math
import statistics
from base64 import b64decode, b64encode
import cv2
import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from fastapi import APIRouter, Request

from ..schemas import TriggerBody
from .capture import get_segment_image, save_json
from .orient import get_mask_multitype, find_closest_stem
from .inspect import get_measures_from_mask
from .utils import CFG_PATH, get_yaml_entry


# Конфигурационные параметры
WITH_DUMPS = get_yaml_entry(CFG_PATH, ['WITH_DUMPS', 'LOCATE'], True)
BENDING_PROPORTION = get_yaml_entry(CFG_PATH, ['BENDING_PROPORTION'], 0.3)
OK_LENGTH_MIN = get_yaml_entry(CFG_PATH, ['OK_LENGTH_MIN'], 5.0)  # см
OK_LENGTH_MAX = get_yaml_entry(CFG_PATH, ['OK_LENGTH_MAX'], 7.0)  # см

router = APIRouter()


def to_pickle(obj: Any, name: str, ext: str = '.pkl') -> None:
    """
    Сохраняет объект в бинарный файл с помощью pickle.

    Args:
        obj: Любой сериализуемый объект.
        name (str): Имя файла (без расширения).
        ext (str): Расширение файла. По умолчанию '.pkl'.
    """
    with open(name + ext, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def from_pickle(name: str, ext: str = '.pkl') -> Optional[Any]:
    """
    Загружает объект из бинарного файла, если он существует.

    Args:
        name (str): Имя файла (без расширения).
        ext (str): Расширение файла.

    Returns:
        Объект или None, если файл не найден.
    """
    if does_exist(name, ext):
        with open(name + ext, 'rb') as f:
            return pickle.load(f)
    return None


def does_exist(name: str, ext: str = '.pkl') -> bool:
    """
    Проверяет существование файла.

    Args:
        name (str): Имя файла.
        ext (str): Расширение.

    Returns:
        bool: True, если файл существует.
    """
    return os.path.exists(name + ext)


def arr_from_b64_str(b64_str: str, shape: Tuple[int, int, int] = (1440, 2560, 3)) -> np.ndarray:
    """
    Декодирует base64-строку в изображение заданной формы.

    Args:
        b64_str (str): Изображение в формате base64.
        shape (tuple): Форма массива (H, W, C). По умолчанию (1440, 2560, 3).

    Returns:
        np.ndarray: Изображение в формате uint8.
    """
    data = np.frombuffer(bytearray(b64decode(b64_str)), dtype=np.uint8)
    return data.reshape(shape)


def get_first_extreme(c: np.ndarray, oper: str) -> List[Tuple[int, int]]:
    """
    Находит первую (по порядку) крайнюю точку по оси.

    Args:
        c (np.ndarray): Контур (N, 2).
        oper (str): 'min' или 'max'.

    Returns:
        list[tuple]: [(y, x)] для min/max по x и y.
    """
    axis = 0 if oper == 'min' else 1
    a = np.argmin(c, axis=0) if oper == 'min' else np.argmax(c, axis=0)
    return [(c[x][1], c[x][0]) for x in a]


def get_middle_extreme(c: np.ndarray, oper: str) -> Tuple[int, int]:
    """
    Находит среднюю из крайних точек (медиану) по оси.

    Полезно для симметричных объектов.

    Args:
        c (np.ndarray): Контур (N, 2).
        oper (str): 'min' или 'max'.

    Returns:
        tuple: (y, x) средней крайней точки.
    """
    if oper == 'min':
        a = np.argwhere(c == c.min(axis=0))
    else:
        a = np.argwhere(c == c.max(axis=0))

    b = [[c[x].tolist() for x, y in a.tolist() if y == r] for r in [0, 1]]
    middle = lambda lst: lst[len(lst) // 2]
    center = lambda lst, k: middle(sorted(lst, key=lambda x: x[k]))
    m = [center(b[i], ii) for i, ii in enumerate([1, 0])]
    return [(x[1], x[0]) for x in m][0]


def get_extremes(
    mask: np.ndarray,
    offset: Tuple[Optional[int], Optional[int], Optional[int], Optional[int]] = (None, 0, 0, 0),
    auto_off: float = 0.85
) -> Tuple[Tuple, Tuple, Tuple, Tuple, Tuple, np.ndarray]:
    """
    Находит крайние точки маски: верх, низ, лево, право и центр.

    Также применяет смещение (offset) для коррекции позиции.

    Args:
        mask (np.ndarray): Бинарная маска объекта.
        offset (tuple): (top, bottom, left, right) смещения. None — автоматическое.
        auto_off (float): Коэффициент автоматического смещения.

    Returns:
        tuple: (top, bottom, left, right, center, mask)
    """
    left, top = get_middle_extreme(mask, 'min')
    right, bottom = get_middle_extreme(mask, 'max')
    tblr = [top, bottom, left, right]
    center = (int(sum(x[0] for x in tblr) / 4), int(sum(x[1] for x in tblr) / 4))

    dist = lambda d1, d2: math.ceil(abs(d2 - d1) * (1 - auto_off))
    off = lambda w: dist(top[0], bottom[0]) if w == 'vert' else dist(left[1], right[1])

    t, b = [x if x is not None else off('vert') for x in offset[:2]]
    l, r = [x if x is not None else off('horiz') for x in offset[2:]]

    top = (top[0] + t, top[1])
    bottom = (bottom[0] - b, bottom[1])
    left = (left[0], left[1] + l)
    right = (right[0], right[1] - r)

    return top, bottom, left, right, center, mask


def get_metrics_coordinates_upper_left(
    u: float, v: float, d: float,
    fx: float = 1498.922607,
    fy: float = 1497.590820,
    cx: float = 1273.266724,
    cy: float = 728.279663
) -> Tuple[float, float, float]:
    """
    Переводит пиксельные координаты (u, v) и глубину d в 3D-координаты (x, y, z).

    Использует модель камеры с верхним левым началом.

    Args:
        u, v: Пиксельные координаты.
        d: Глубина (z) в мм.
        fx, fy: Фокусные расстояния.
        cx, cy: Главная точка.

    Returns:
        tuple: (x, y, z) в мм.
    """
    z = d
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return x, y, z


def get_metrics_coordinates(
    u: float, v: float, d: float, **kwargs
) -> Tuple[float, float, float, float]:
    """
    Переводит пиксельные координаты в метрические относительно центра изображения.

    Args:
        u, v: Пиксельные координаты.
        d: Глубина.
        **kwargs: Параметры камеры.

    Returns:
        tuple: (x_, y_, d, h) — смещение от центра и расстояние до камеры.
    """
    x, y, _ = get_metrics_coordinates_upper_left(u, v, d, **kwargs)
    x0, y0, _ = get_metrics_coordinates_upper_left(1440 / 2, 2560 / 2, d, **kwargs)
    x_, y_ = y - y0, x0 - x
    h = math.sqrt(x_ ** 2 + y_ ** 2 + d ** 2)
    return x_, y_, d, h


def is_shape_valid(mask: Any, active: bool = True) -> bool:
    """
    Проверяет, соответствует ли форма маски ожидаемым пропорциям (например, огурцу).

    Использует эллипс для оценки соотношения сторон.

    Args:
        mask: Маска (может быть torch.Tensor).
        active (bool): Включена ли проверка.

    Returns:
        bool: True, если форма допустима.
    """
    if not active:
        return True

    try:
        mask_np = mask.cpu().numpy().astype(np.uint8)
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
        cnt = max(contours, key=cv2.contourArea)
        (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
        a, b = sorted([MA, ma], reverse=True)
        ratio = a / b
        if not 2.5 <= ratio <= 4.5:
            print(f'Dimensions not valid!! MA {a} ma {b} proportion {ratio}')
            return False
        return True
    except Exception as e:
        print(f'Shape validation error: {e}')
        return False


def get_coordinates(
    annotations: List[Any],
    depth_data: np.ndarray,
    filter_zeroes: bool = True,
    filter_outliers: bool = True
) -> Tuple[
    List[Dict],                    # Пиксельные координаты
    List[Dict],                    # Метрические (относительно камеры)
    List[Dict]                     # Метрические (относительно центра)
]:
    """
    Извлекает координаты крайних точек огурцов из аннотаций и данных глубины.

    Также определяет:
    - Длину, изгиб, угол.
    - Привязку к стеблю.
    - Классификацию (годен/нет).

    Args:
        annotations (list): Результаты YOLO (с масками).
        depth_data (np.ndarray): Массив глубины (H, W).
        filter_zeroes (bool): Заменять нулевую глубину на медиану.
        filter_outliers (bool): Корректировать выбросы глубины.

    Returns:
        tuple: (pixel_coords, metric_coords, centered_metric_coords)
    """

    def filter_depth(
        d_val: float,
        point_list: List[Tuple[int, int]],
        f_zeroes: bool,
        f_outliers: bool,
        radius: int = 100
    ) -> float:
        """Фильтрует значение глубины с учётом соседних точек."""
        res = d_val
        new_p_list = [(x, y, depth_data[x - 1][y - 1]) for x, y in point_list]
        trimmed_p_list = [(x, y, z) for x, y, z in new_p_list if z != 0]
        z_list = [z for x, y, z in trimmed_p_list]

        if not z_list:
            return res

        median_z = statistics.median(z_list)

        if f_zeroes and res == 0 and median_z > 0:
            res = median_z

        if f_outliers:
            far, near = median_z + radius, median_z - radius
            if res > far or res < near:
                res = median_z

        return res

    results = []
    for ix, annotation in enumerate(annotations):
        try:
            cucumber_ixs = [i for i in range(len(annotation.boxes.cls)) if int(annotation.boxes.cls[i]) == 0]
            stem_ixs = [i for i in range(len(annotation.boxes.cls)) if int(annotation.boxes.cls[i]) == 1]

            stems_per_cucumber = None
            if annotation.masks:
                cucumber_mask_objs = [annotation.masks.data[i] for i in cucumber_ixs]
                stem_mask_objs = [annotation.masks.data[i] for i in stem_ixs]
                orig_img = cv2.cvtColor(annotation.orig_img, cv2.COLOR_RGB2BGR)
                cucumber_masks = [get_mask_multitype(x, orig_img, 0)[0] for x in cucumber_mask_objs]
                stem_masks = [get_mask_multitype(x, orig_img, 1)[0] for x in stem_mask_objs]
                stems_per_cucumber = [find_closest_stem(c_mask, i, stem_masks, orig_img) for i, c_mask in enumerate(cucumber_masks)]

            masks = annotation.masks.xy if annotation.masks else []
            raw_masks = annotation.masks.data if annotation.masks else []

            for i, mask in enumerate(masks):
                if i in cucumber_ixs and is_shape_valid(raw_masks[i]):
                    cucumber_stem_info = stems_per_cucumber[cucumber_ixs.index(i)] if stems_per_cucumber else None
                    mask_image = mask.astype(int)
                    top, bottom, left, right, center, _ = get_extremes(mask_image, (None, 0, 0, 0))

                    points = {'top': top, 'bottom': bottom, 'left': left, 'right': right, 'center': center}
                    res_dict = {
                        k: (
                            int(v[0]), int(v[1]),
                            int(filter_depth(depth_data[v[0] - 1][v[1] - 1], points.values(), filter_zeroes, filter_outliers))
                        ) for k, v in points.items()
                    }

                    if cucumber_stem_info:
                        res_dict['side'] = cucumber_stem_info['side']

                    # Получаем измерения
                    measures = get_measures_from_mask(raw_masks[i], (res_dict['top'][2], res_dict['bottom'][2]))
                    if measures:
                        res_dict['arc_length'] = measures['length_mm']
                        res_dict['line_length'] = measures['straight_line_mm']
                        res_dict['bending'] = measures['depth_mm']
                        bending_ok = measures['depth_mm'] / (measures['straight_line_mm'] + 1e-6) < BENDING_PROPORTION
                        is_small = measures['straight_line_mm'] < OK_LENGTH_MIN
                        is_big = measures['straight_line_mm'] > OK_LENGTH_MAX
                        suitable = not (is_small or is_big)
                        to_take = not is_small

                        res_dict.update({
                            'is_good': suitable,
                            'to_take': to_take,
                            'angle': measures['line_to_y_angle_deg']
                        })

                    results.append(res_dict)

        except Exception as e:
            print(f'Error processing annotation {ix}: {e}')

    if not results:
        return [], [], []

    # Сортировка по глубине (ближайшие — первыми)
    sorted_results = sorted(results, key=lambda d: d['center'][2])
    cols = ['top', 'bottom', 'left', 'right', 'center']

    # Метрические координаты
    metric_raw = [
        {k: v if k not in cols else get_metrics_coordinates_upper_left(v[0], v[1], v[2]) for k, v in x.items()}
        for x in sorted_results
    ]
    metric_centered = [
        {k: v if k not in cols else get_metrics_coordinates(v[0], v[1], v[2]) for k, v in x.items()}
        for x in sorted_results
    ]

    return sorted_results, metric_raw, metric_centered


async def write_results(image_data: np.ndarray, results: Dict, output_folder: str) -> None:
    """
    Асинхронно сохраняет результаты и изображение.

    Args:
        image_data (np.ndarray): Изображение.
        results (dict): Результаты анализа.
        output_folder (str): Путь для сохранения.
    """
    json_path = os.path.join(output_folder, "image_coordinates_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    cv2.imwrite(os.path.join(output_folder, "image_distance_ranked.jpg"), image_data)


def dump_results(output_folder: str, image_data: np.ndarray, coordinates: Dict) -> Dict:
    """
    Формирует и сохраняет результаты с base64-изображением.

    Args:
        output_folder (str): Каталог вывода.
        image_data (np.ndarray): Изображение.
        coordinates (dict): Координаты.

    Returns:
        dict: Результаты с base64.
    """
    b64image = b64encode(image_data.tobytes()).decode('utf-8')
    results = {
        'image': b64image,
        'coordinates': coordinates['pixel_coordinates'],
        'all_coordinates': coordinates
    }
    if WITH_DUMPS:
        asyncio.run(write_results(image_data, results, output_folder))
    return results


def put_labels(image: np.ndarray, coordinates: List[Dict]) -> np.ndarray:
    """
    Наносит номера на изображение по центрам огурцов.

    Args:
        image (np.ndarray): Исходное изображение.
        coordinates (list): Список координат.

    Returns:
        np.ndarray: Изображение с метками.
    """
    new_image = image.copy()
    for i, coord in enumerate(coordinates):
        yx = tuple(coord['center'][:2])
        xy = (yx[1], yx[0])
        cv2.putText(new_image, str(i), xy, cv2.FONT_HERSHEY_DUPLEX, 6.0, (255, 255, 255), 20)
        cv2.putText(new_image, str(i), xy, cv2.FONT_HERSHEY_DUPLEX, 6.0, (0, 0, 0), 5)
    return new_image


@router.post("/capture_segment_locate_default")
def get_segment_locate_image(request: Request, body: TriggerBody) -> Dict:
    """
    Эндпоинт: захват → сегментация → локализация огурцов.

    Выполняет:
    1. Захват изображения с камеры.
    2. Сегментацию (YOLO).
    3. Извлечение координат, длины, изгиба, угла.
    4. Классификацию (годен/не годен).
    5. Визуализацию и сохранение.

    Args:
        request (Request): FastAPI-запрос (для доступа к состоянию).
        body (TriggerBody): Параметры (папка, яркость, уверенность).

    Returns:
        dict: Результаты с изображением, координатами и временем.
    """
    tic = time.perf_counter()
    captures, segments, capture_segment_duration = get_segment_image(request=request, body=body)
    tic1 = time.perf_counter()

    # Извлечение координат
    coordinates, metric_coordinates, metric_centered = get_coordinates(
        annotations=segments[1],  # Второй элемент — результаты сегментации
        depth_data=captures['depth_data']
    )

    output_folder = body.output_folder
    new_image = arr_from_b64_str(segments[0]['image'])  # Базовое изображение
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)

    if coordinates:
        new_image = put_labels(new_image, coordinates)

    # Добавление порядкового номера
    labeled_coordinates = [{"order": i, **c} for i, c in enumerate(coordinates)]
    labeled_metric = [{"order": i, **c} for i, c in enumerate(metric_coordinates)]
    labeled_centered = [{"order": i, **c} for i, c in enumerate(metric_centered)]

    all_coords = {
        "pixel_coordinates": labeled_coordinates,
        "metric_coordinates": labeled_metric,
        "metric_coordinates_centered": labeled_centered
    }

    results = dump_results(output_folder, new_image, all_coords)

    toc = time.perf_counter()
    duration = {
        **capture_segment_duration,
        'locate': toc - tic1,
        'capture_detect_segment_locate': toc - tic
    }

    return {**results, "duration": duration}
