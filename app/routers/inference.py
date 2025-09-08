import asyncio
import time
import os
import pickle
from fastapi import APIRouter, Request
from base64 import b64decode, b64encode
import json
import numpy as np
import cv2
from ..schemas import SegmentBody
from .utils import CFG_PATH, get_yaml_entry


# Флаг: сохранять ли промежуточные данные (изображения, аннотации) на диск
WITH_DUMPS = get_yaml_entry(CFG_PATH, ['WITH_DUMPS', 'INFERENCE'], False)

router = APIRouter()


def predict(chosen_model, img, **kwargs):
    """
    Выполняет предсказание с помощью модели YOLO.

    Обёртка вокруг метода `.predict()` модели Ultralytics.

    Args:
        chosen_model: Загруженная модель YOLO (например, `YOLO('yolov8n-seg.pt')`).
        img (np.ndarray): Входное изображение в формате BGR.
        **kwargs: Дополнительные параметры (например, `conf`, `imgsz`, `save` и т.д.).

    Returns:
        list: Список объектов `Results` из Ultralytics.
    """
    return chosen_model.predict(img, **kwargs)


def predict_and_detect(chosen_model, img, **kwargs):
    """
    Выполняет предсказание и накладывает bounding box аннотации на изображение.

    Для каждого обнаруженного объекта:
    - Рисует прямоугольник.
    - Добавляет подпись с именем класса.

    Args:
        chosen_model: Модель YOLO.
        img (np.ndarray): Исходное изображение (BGR).
        **kwargs: Параметры для `.predict()`.

    Returns:
        tuple[np.ndarray, list]: 
            - Изображение с аннотациями (не изменяет оригинальное).
            - Список результатов детекции.
    """
    results = predict(chosen_model, img, **kwargs)

    for result in results:
        for box in result.boxes:
            # Координаты bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = result.names[cls_id]

            # Рисуем прямоугольник и подпись
            cv2.rectangle(result.orig_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(result.orig_img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)

    return img, results


async def to_pickle(obj, name: str, ext: str = '.pkl'):
    """
    Асинхронно сохраняет объект в бинарный файл с помощью pickle.

    Args:
        obj: Любой сериализуемый Python-объект.
        name (str): Имя файла (без расширения).
        ext (str): Расширение файла. По умолчанию '.pkl'.
    """
    with open(name + ext, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


async def write_image(image_data: np.ndarray, filename: str):
    """
    Асинхронно сохраняет изображение на диск с помощью OpenCV.

    Обёртка для `cv2.imwrite`, запускаемая через asyncio.

    Args:
        image_data (np.ndarray): Данные изображения (BGR).
        filename (str): Путь для сохранения.
    """
    cv2.imwrite(filename, image_data)


def arr_from_b64_str(b64_str: str, shape: tuple = (1440, 2560, 3)) -> np.ndarray:
    """
    Декодирует base64-строку в NumPy-массив заданной формы.

    Используется для преобразования изображений, переданных в base64.

    Args:
        b64_str (str): Строка в формате base64.
        shape (tuple): Ожидаемая форма массива (высота, ширина, каналы). По умолчанию (1440, 2560, 3).

    Returns:
        np.ndarray: Изображение в формате uint8.

    Raises:
        ValueError: Если данные не соответствуют заданной форме.
    """
    try:
        decoded = np.frombuffer(bytearray(b64decode(b64_str)), dtype=np.uint8)
        return decoded.reshape(shape)
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {e}")


def get_dump_results(
    image_data: np.ndarray,
    image_path: str,
    width: int,
    height: int,
    annotations,
    output_folder: str
) -> dict:
    """
    Формирует результаты сегментации для возврата и, при необходимости, сохраняет их на диск.

    Включает:
    - base64-представление аннотированного изображения.
    - путь к файлу.
    - аннотации (результаты YOLO).
    - метаданные (размеры).

    Args:
        image_data (np.ndarray): Аннотированное изображение.
        image_path (str): Путь к сохранённому файлу.
        width (int): Ширина изображения.
        height (int): Высота изображения.
        annotations: Результаты модели (объекты `Results` из Ultralytics).
        output_folder (str): Каталог для сохранения дампов.

    Returns:
        dict: Структура с результатами, готовая для JSON-сериализации.
    """
    if WITH_DUMPS:
        asyncio.run(to_pickle(image_data, os.path.join(output_folder, 'image_data')))

    b64image = b64encode(image_data.tobytes()).decode('utf-8')

    results = {
        "image": b64image,
        "image_path": image_path,
        "annotations": str(annotations),  # Простая строка; можно улучшить при необходимости
        "box_format": "xyxy",
        "img_width": width,
        "img_height": height,
    }

    if WITH_DUMPS:
        asyncio.run(to_pickle(results, os.path.join(output_folder, 'annotation_segmentation_results')))

    return results


def get_segments_visual(results) -> np.ndarray:
    """
    Накладывает маски сегментации на изображение.

    Для каждого сегмента:
    - Использует контуры из `.masks.xy`.
    - Рисует их зелёным цветом с полной заливкой.

    Args:
        results: Список объектов `Results` из Ultralytics (с атрибутом `.masks`).

    Returns:
        np.ndarray: Изображение с наложенными масками сегментации.
                    Возвращает оригинальное изображение, если масок нет.
    """
    background = None
    for result in results:
        background = result.orig_img
        try:
            masks = result.masks.xy  # Список полигонов
        except AttributeError:
            masks = []
            print('No segmentation masks found.')

        for mask in masks:
            mask = mask.astype(np.int32)
            cv2.drawContours(background, [mask], -1, (0, 255, 0), thickness=cv2.FILLED)

    return background if background is not None else np.zeros((480, 640, 3), dtype=np.uint8)


@router.post("/segment")
def segment_image(request: Request, body: SegmentBody):
    """
    Эндпоинт для сегментации изображения с помощью модели YOLO.

    Поддерживает:
    - Загрузку изображения через base64 (`body.image`).
    - Загрузку по пути (`body.image_path`).
    - Настройку уверенности (`confidence`).
    - Сохранение результатов (если `WITH_DUMPS=True`).

    Этапы:
    1. Загрузка изображения.
    2. Детекция и рисование bounding box.
    3. Наложение масок сегментации.
    4. Формирование результата с base64 и метаданными.
    5. Возврат результата и времени выполнения.

    Args:
        request (Request): Объект запроса FastAPI (для доступа к модели и состоянию).
        body (SegmentBody): Тело запроса с параметрами сегментации.

    Returns:
        tuple[dict, dict]:
            - Результаты сегментации (изображение, аннотации, пути).
            - Время выполнения этапов:
                - 'detect': детекция
                - 'segment': сегментация
                - 'detect_segment': общее время
    """
    tic1 = time.perf_counter()

    confidence = body.confidence
    output_folder = body.output_folder

    # Загрузка изображения
    if body.image is not None:
        image = arr_from_b64_str(body.image)
    else:
        image = cv2.imread(body.image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {body.image_path}")

    # Загрузка модели
    target_model = request.app.state.ml_models["yolo"]

    # Параметры предсказания
    params = {
        'conf': confidence,
        'imgsz': 1280,
        'save': False,
        'show': False
    }

    task = request.app.state.task
    print(f"The task is {task}")

    # Выполнение детекции и рисование bounding box
    new_image, results = predict_and_detect(target_model, image, **params)

    # Сохранение аннотированного изображения
    if WITH_DUMPS:
        asyncio.run(write_image(new_image, os.path.join(output_folder, "annotated_image.jpg")))

    toc1 = time.perf_counter()
    duration = {'detect': toc1 - tic1}
    tic2 = time.perf_counter()

    # Наложение масок сегментации
    annotated_frame = get_segments_visual(results)
    if annotated_frame is None:
        annotated_frame = image
        print("Image annotation unsuccessful")

    annotated_output_path = os.path.join(output_folder, "annotated_segmented_image.jpg")
    if WITH_DUMPS:
        asyncio.run(write_image(annotated_frame, annotated_output_path))

    # Определение размеров
    try:
        width, height = annotated_frame.size  # PIL Image
    except AttributeError:
        height, width = annotated_frame.shape[:2]  # OpenCV

    # Формирование итоговых результатов
    results_dump = get_dump_results(
        image_data=annotated_frame,
        image_path=annotated_output_path,
        width=width,
        height=height,
        annotations=results,
        output_folder=output_folder
    )

    toc2 = time.perf_counter()
    duration.update({
        'segment': toc2 - tic2,
        'detect_segment': toc2 - tic1
    })

    return results_dump, duration
