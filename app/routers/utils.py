import yaml
import numpy as np
import cv2
from PIL import Image
from typing import Any, Dict, List, Union, Optional

# Путь к основному конфигурационному файлу
CFG_PATH = '/home/jetadmin/Apps/Cucumber/Server/Loop/app/routers/config.yaml'


def resize_image_to_match(
    reference_image_array: np.ndarray,
    target_image_path: str
) -> np.ndarray:
    """
    Изменяет размер изображения по пути так, чтобы оно соответствовало размерам эталонного изображения.

    Использует OpenCV как основной метод, PIL — как резервный.

    Args:
        reference_image_array (np.ndarray): Эталонное изображение (определяет размер и количество каналов).
        target_image_path (str): Путь к изображению, которое нужно изменить.

    Returns:
        np.ndarray: Изображение в формате numpy, соответствующее по размеру и цветовому пространству.

    Raises:
        Exception: Если ни один из методов не смог загрузить или обработать изображение.
    """
    # Определение целевых размеров
    if reference_image_array.ndim == 3:
        target_height, target_width, _ = reference_image_array.shape
    else:
        target_height, target_width = reference_image_array.shape

    # --- Попытка 1: OpenCV ---
    try:
        img = cv2.imread(target_image_path)
        if img is None:
            raise ValueError(f"Failed to load image with OpenCV: {target_image_path}")

        # Адаптация каналов
        ref_is_color = len(reference_image_array.shape) == 3
        img_is_color = len(img.shape) == 3

        if not img_is_color and ref_is_color:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img_is_color and not ref_is_color:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Изменение размера
        resized = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

        # Конвертация BGR → RGB, если необходимо
        if ref_is_color and resized.ndim == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        return resized

    except Exception as e:
        print(f"OpenCV method failed: {e}. Falling back to PIL...")

    # --- Попытка 2: PIL ---
    try:
        img = Image.open(target_image_path)
        resized_img = img.resize((target_width, target_height), Image.LANCZOS)
        resized_array = np.array(resized_img)

        # Адаптация каналов
        ref_is_color = len(reference_image_array.shape) == 3
        arr_is_color = len(resized_array.shape) == 3

        if not arr_is_color and ref_is_color:
            # Градации серого → RGB
            resized_array = np.stack([resized_array] * 3, axis=-1)
        elif arr_is_color and not ref_is_color:
            # RGB → градации серого
            resized_array = np.mean(resized_array[:, :, :3], axis=2).astype(resized_array.dtype)

        return resized_array

    except Exception as e:
        raise Exception(f"Both OpenCV and PIL resize methods failed for {target_image_path}: {e}")


# --- YAML-утилиты ---

def read_yaml(filepath: str) -> Dict[str, Any]:
    """
    Читает YAML-файл и возвращает содержимое в виде словаря.

    Args:
        filepath (str): Путь к YAML-файлу.

    Returns:
        dict: Данные из файла или пустой словарь при ошибке/отсутствии данных.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            return data if data is not None else {}
    except FileNotFoundError:
        print(f"Config file not found: {filepath}. Returning empty dict.")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {filepath}: {e}")
        return {}
    except Exception as e:
        print(f"Unexpected error reading {filepath}: {e}")
        return {}


def write_yaml(filepath: str, data: Dict[str, Any]) -> None:
    """
    Записывает словарь в YAML-файл.

    Args:
        filepath (str): Путь к файлу.
        data (dict): Данные для записи.
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as file:
            yaml.dump(data, file, default_flow_style=False, sort_keys=False, indent=2)
    except Exception as e:
        raise Exception(f"Failed to write YAML file {filepath}: {e}")


def get_yaml_entry(
    filepath: str,
    key_path: List[Union[str, int]],
    default: Any = None
) -> Any:
    """
    Получает значение по вложенному ключу из YAML-файла.

    Пример: get_yaml_entry('config.yaml', ['model', 'name']) → data['model']['name']

    Args:
        filepath (str): Путь к YAML-файлу.
        key_path (list): Список ключей для доступа к вложенному значению.
        default (Any): Значение по умолчанию, если ключ не найден.

    Returns:
        Any: Значение по ключу или значение по умолчанию.
    """
    data = read_yaml(filepath)
    current = data

    for key in key_path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default

    return current


def set_yaml_entry(
    filepath: str,
    key_path: List[Union[str, int]],
    value: Any
) -> None:
    """
    Устанавливает значение по вложенному ключу в YAML-файле.

    Автоматически создаёт промежуточные словари при необходимости.

    Пример: set_yaml_entry('config.yaml', ['server', 'port'], 8080)

    Args:
        filepath (str): Путь к YAML-файлу.
        key_path (list): Список ключей (включая вложенные).
        value (Any): Значение для установки.
    """
    data = read_yaml(filepath)
    current = data

    for i, key in enumerate(key_path):
        if i == len(key_path) - 1:
            current[key] = value
        else:
            if not isinstance(current, dict):
                current = {}
            if key not in current:
                current[key] = {}
            current = current[key]

    write_yaml(filepath, data)
