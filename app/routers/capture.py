import asyncio
import time
from base64 import b64encode
from fastapi import APIRouter, Request
from pyorbbecsdk import *
import cv2
import os
import json
import numpy as np
from typing import Union, Any, Optional
from ..schemas import SegmentBody, TriggerBody
from .inference import segment_image
from .utils import CFG_PATH, get_yaml_entry, resize_image_to_match


# Конфигурация: сохранять ли дампы (изображения, JSON) на диск
WITH_DUMPS = get_yaml_entry(CFG_PATH, ['WITH_DUMPS', 'CAPTURE'], True)

# Таймаут ожидания кадров от камеры (в миллисекундах)
WAIT_TIMEOUT = get_yaml_entry(CFG_PATH, ['FRAME_WAIT_TIMEOUT'], 100)


router = APIRouter()


def frame_to_bgr_image(frame: VideoFrame) -> Union[Optional[np.ndarray], Any]:
    """
    Конвертирует видеокадр Orbbec в BGR-изображение (формат OpenCV).

    Поддерживает различные форматы цвета: RGB, BGR, MJPG, YUYV, NV12, NV21, I420, UYVY.
    Для сжатых или полу-планарных форматов применяются соответствующие методы декодирования.

    Args:
        frame (VideoFrame): Входной видеокадр от камеры Orbbec.

    Returns:
        np.ndarray: Изображение в формате BGR (OpenCV), или None при ошибке.

    Notes:
        - Для MJPG используется cv2.imdecode.
        - Для NV12/NV21/I420 требуется ручная конвертация.
    """
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())
    image = np.zeros((height, width, 3), dtype=np.uint8)

    if color_format == OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif color_format == OBFormat.BGR:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_format == OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    elif color_format == OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif color_format == OBFormat.I420:
        image = i420_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.NV12:
        image = nv12_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.NV21:
        image = nv21_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.UYVY:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    else:
        print(f"Unsupported color format: {color_format}")
        return None
    return image


async def write_image(image_data: np.ndarray, filename: str):
    """
    Асинхронно сохраняет изображение на диск с использованием OpenCV.

    Args:
        image_data (np.ndarray): Данные изображения.
        filename (str): Путь для сохранения файла.
    """
    cv2.imwrite(filename, image_data)


def save_image(
    image_folder: str,
    image_data: np.ndarray,
    image_name: str,
    image_type: str,
    ext: str = 'jpg',
    force: bool = False
) -> str:
    """
    Сохраняет изображение в указанный каталог, если разрешено конфигурацией.

    Имя файла формируется по шаблону: `{image_type}_{image_name}_{width}x{height}.jpg`.

    Args:
        image_folder (str): Каталог для сохранения.
        image_data (np.ndarray): Данные изображения.
        image_name (str): Базовое имя файла.
        image_type (str): Тип изображения (например, "color", "depth").
        ext (str): Расширение файла. По умолчанию 'jpg'.
        force (bool): Принудительно сохранить, даже если WITH_DUMPS=False.

    Returns:
        str: Полный путь к сохранённому файлу.
    """
    filename = os.path.join(
        image_folder,
        f'{image_type}_{image_name}_{image_data.shape[1]}x{image_data.shape[0]}.{ext}'
    )
    if WITH_DUMPS or force:
        asyncio.run(write_image(image_data, filename))
    return filename


def save_json(results: dict, output_folder: str, fname: str):
    """
    Сохраняет словарь в формате JSON на диск.

    Args:
        results (dict): Данные для сохранения.
        output_folder (str): Каталог назначения.
        fname (str): Имя файла.
    """
    with open(os.path.join(output_folder, fname), "w") as f:
        json.dump(results, f, indent=4)


def get_dump_results(
    color_image_path: str,
    width: int,
    height: int,
    depth_image_path: str,
    brightened_image_path: str,
    depth_data: np.ndarray,
    output_folder: str
) -> dict:
    """
    Формирует словарь с результатами захвата изображений.

    При включённой опции WITH_DUMPS добавляет base64-представление данных глубины в JSON.

    Args:
        color_image_path (str): Путь к цветному изображению.
        width (int): Ширина кадра.
        height (int): Высота кадра.
        depth_image_path (str): Путь к изображению глубины.
        brightened_image_path (str): Путь к яркостно-скорректированному изображению.
        depth_data (np.ndarray): Массив данных глубины (в метрах).
        output_folder (str): Каталог для сохранения JSON.

    Returns:
        dict: Результаты захвата с метаданными и, при необходимости, base64-данными.
    """
    results = {
        "color_image_path": color_image_path,
        "depth_image_path": depth_image_path,
        "brightened_image_path": brightened_image_path,
        "width": width,
        "height": height
    }

    if WITH_DUMPS:
        print(f'Depth data shape: {depth_data.shape}, type: {depth_data.dtype}')
        b64image = b64encode(depth_data.tobytes()).decode('utf-8')
        results["depth_data"] = b64image
        save_json(results, output_folder, "image_capture_results.json")

    results["depth_data"] = depth_data  # Возвращаем как numpy массив для дальнейшей обработки
    return results


def get_image_from_cam(
    pipeline: Pipeline,
    output_folder: str,
    brighten: int
) -> tuple[dict, float]:
    """
    Захватывает синхронизированные цветной и глубинный кадры с камеры Orbbec.

    - Выполняет выравнивание потоков (по цвету).
    - Конвертирует кадры в OpenCV-изображения.
    - Сохраняет цветное, глубинное, комбинированное и яркостно-скорректированное изображения.
    - Возвращает метаданные и время выполнения.

    Args:
        pipeline (Pipeline): Активный конвейер камеры Orbbec.
        output_folder (str): Каталог для сохранения изображений.
        brighten (int): Значение яркости для увеличения (в HSV-канале V).

    Returns:
        tuple[dict, float]: 
            - Словарь с путями к файлам и данными глубины.
            - Время захвата в секундах.
    """
    tic = time.perf_counter()
    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    while True:
        frames = pipeline.wait_for_frames(WAIT_TIMEOUT)
        if not frames:
            continue

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        frames = align_filter.process(frames)
        if not frames:
            continue

        frames = frames.as_frame_set()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_image = frame_to_bgr_image(color_frame)
        use_dummy = get_yaml_entry(CFG_PATH, ['USE_DUMMY'], False)
        if use_dummy:
            dummy_path = get_yaml_entry(CFG_PATH, ['DUMMY_PATH'])
            if dummy_path:
                try:
                    color_image = resize_image_to_match(color_image, dummy_path)
                except Exception as e:
                    print(f"Failed to apply dummy image resize: {e}")

        if color_image is None:
            print("Failed to convert frame to image")
            continue

        break

    width = depth_frame.get_width()
    height = depth_frame.get_height()
    scale = depth_frame.get_depth_scale()

    color_image_reduced = color_image  # Можно добавить resize при необходимости
    color_image_path = save_image(
        image_folder=output_folder,
        image_data=color_image_reduced,
        image_name="image",
        image_type="color"
    )

    # Обработка данных глубины
    depth_data_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
    depth_data = depth_data_raw.reshape((height, width))
    depth_data_scaled = depth_data.astype(np.float32) * scale  # в метрах

    print(f"Depth data scaled shape: {depth_data_scaled.shape}")

    depth_image_raw = cv2.normalize(depth_data_scaled, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_image = cv2.applyColorMap(depth_image_raw, cv2.COLORMAP_JET)
    depth_image_path = save_image(
        image_folder=output_folder,
        image_data=depth_image,
        image_name="image",
        image_type="depth",
        force=False
    )

    # Комбинированное изображение (цвет + глубина)
    weighted_image = cv2.addWeighted(color_image_reduced, 0.5, depth_image, 0.5, 0)
    combined_image_path = save_image(
        image_folder=output_folder,
        image_data=weighted_image,
        image_name="image",
        image_type="combined",
        force=False
    )

    # Яркостная коррекция
    hsv = cv2.cvtColor(color_image_reduced, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.add(hsv[:, :, 2], brighten)
    brightened_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    brightened_image_path = save_image(
        image_folder=output_folder,
        image_data=brightened_image,
        image_name="image",
        image_type="brightened",
        force=True
    )

    results = get_dump_results(
        color_image_path=color_image_path,
        width=width,
        height=height,
        depth_image_path=depth_image_path,
        brightened_image_path=brightened_image_path,
        depth_data=depth_data_scaled,
        output_folder=output_folder
    )

    toc = time.perf_counter()
    return results, toc - tic


@router.post("/capture_segment")
def get_segment_image(request: Request, body: TriggerBody):
    """
    Эндпоинт для захвата кадра с камеры и его сегментации.

    Последовательность:
    1. Захват цветного и глубинного кадров.
    2. Обработка: выравнивание, коррекция яркости.
    3. Сохранение промежуточных изображений.
    4. Запуск сегментации через YOLO-модель.
    5. Возврат результатов и времени выполнения.

    Args:
        request (Request): Объект запроса FastAPI (для доступа к состоянию приложения).
        body (TriggerBody): Параметры запроса (папка вывода, яркость, уверенность и т.д.).

    Returns:
        dict: Словарь с:
            - Результатами захвата (пути к файлам, данные глубины).
            - Результатами сегментации (маски, bounding boxes).
            - Временем выполнения этапов (захват, сегментация, общее).
    """
    tic = time.perf_counter()
    pipeline = request.app.state.cam_pipelines["pipeline"]

    # Шаг 1: Захват изображения
    captures, capture_seconds = get_image_from_cam(pipeline, body.output_folder, body.brighten)

    # Шаг 2: Сегментация яркостно-скорректированного изображения
    segments, segment_duration = segment_image(
        request=request,
        body=SegmentBody(
            image_path=captures["brightened_image_path"],
            confidence=body.confidence,
            output_folder=body.output_folder
        )
    )

    toc = time.perf_counter()
    duration = {
        'capture': capture_seconds,
        **segment_duration,
        'capture_detect_segment': toc - tic
    }

    return captures, segments, duration
