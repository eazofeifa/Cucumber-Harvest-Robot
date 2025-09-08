import sys
sys.path.append("/home/jetadmin/Apps/Cucumber/Capture/pyorbbecsdk/install/lib/")

from fastapi import FastAPI
from contextlib import asynccontextmanager
import torch
from ultralytics import YOLO
from pyorbbecsdk import *
from app.routers import inference, capture, locate, loop, configurate, inspect
from app.routers.utils import CFG_PATH, get_yaml_entry


# Константы приложения
YOLO_CHECKPOINT_PATH = f"/home/jetadmin/Apps/Cucumber/Inference/Weights/{get_yaml_entry(CFG_PATH, ['MODEL_NAME'], 'segment-cucumber-orig.pt')}"
"""
Путь к файлу весов модели YOLO. Значение берётся из конфигурационного файла (CFG_PATH),
а в случае отсутствия — используется значение по умолчанию 'segment-cucumber-orig.pt'.
"""

ALIGN_MODE = 'SW'
"""
Режим выравнивания потоков с камер: 
'SW' — программное выравнивание, 
'HW' — аппаратное, 
любое другое значение — выравнивание отключено.
"""

ENABLE_SYNC = True
"""
Флаг, определяющий, нужно ли синхронизировать фреймы с разных сенсоров (цвет и глубина).
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Асинхронный контекстный менеджер для инициализации и очистки ресурсов при запуске и остановке приложения.

    При старте:
    - Загружает модель YOLO и помещает её в состояние приложения.
    - Настраивает и запускает конвейер камеры Orbbec (цвет + глубина) с заданными параметрами разрешения и частоты.
    - Включает синхронизацию фреймов и режим выравнивания (аппаратный/программный).
    
    При завершении:
    - Останавливает конвейер камеры.
    - Очищает состояние модели и конвейера.

    Args:
        app (FastAPI): Экземпляр приложения FastAPI, в состояние которого добавляются ресурсы.

    Yields:
        None: Управление передаётся приложению на время его работы.
    """
    app.state.ml_models = {}
    app.state.cam_pipelines = {}
    
    # Определение устройства для модели YOLO
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Определение задачи модели (segment, если расширение не .pt)
    task = None if '.pt' in YOLO_CHECKPOINT_PATH else 'segment'
    
    # Загрузка модели YOLO
    yolo_model = YOLO(YOLO_CHECKPOINT_PATH) if not task else YOLO(YOLO_CHECKPOINT_PATH, task=task)
    
    try:
        yolo_model.to(device=device)
        app.state.task = None
    except Exception:
        # Если не удалось загрузить на устройство, использовать сегментацию
        app.state.task = 'segment'
    
    app.state.ml_models["yolo"] = yolo_model
    
    # Настройка камеры Orbbec
    config = Config()
    pipeline = Pipeline()
    device = pipeline.get_device()
    device_info = device.get_device_info()
    device_pid = device_info.get_pid()  # Можно использовать для логирования или идентификации

    # Настройка цветового потока: 2560x1440, MJPG, 15 FPS
    color_profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = color_profile_list.get_video_stream_profile(2560, 1440, OBFormat.MJPG, 15)
    config.enable_stream(color_profile)

    # Настройка потока глубины: 1024x1024, Y16, 15 FPS
    depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = depth_profile_list.get_video_stream_profile(1024, 1024, OBFormat.Y16, 15)
    config.enable_stream(depth_profile)

    # Установка режима выравнивания цвета и глубины
    if ALIGN_MODE == 'HW':
        config.set_align_mode(OBAlignMode.HW_MODE)
    elif ALIGN_MODE == 'SW':
        config.set_align_mode(OBAlignMode.SW_MODE)
    else:
        config.set_align_mode(OBAlignMode.DISABLE)

    # Включение синхронизации фреймов, если требуется
    if ENABLE_SYNC:
        pipeline.enable_frame_sync()

    # Запуск конвейера
    pipeline.start(config)
    app.state.cam_pipelines["pipeline"] = pipeline

    yield  # Передача управления приложению

    # Очистка ресурсов при завершении
    app.state.ml_models.clear()
    app.state.cam_pipelines["pipeline"].stop()
    app.state.cam_pipelines.clear()


# Инициализация FastAPI приложения с управлением жизненным циклом
app = FastAPI(lifespan=lifespan)


# Подключение маршрутов
app.include_router(configurate.router)
app.include_router(inference.router)
app.include_router(capture.router)
app.include_router(locate.router)
app.include_router(loop.router)
app.include_router(inspect.router)


@app.get("/healthcheck")
def check_working():
    """
    Маршрут для проверки работоспособности сервиса.

    Используется для healthcheck в системах мониторинга или оркестрации (например, Docker, Kubernetes).

    Returns:
        dict: Словарь с признаком доступности и приветственным сообщением.
              Пример: {"online": True, "message": "Hello World!"}
    """
    return {"online": True, "message": "Hello World!"}
