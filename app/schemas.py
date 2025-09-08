from pydantic import BaseModel
from typing import List, Optional


class SegmentBody(BaseModel):
    """
    Модель запроса для выполнения сегментации изображения с помощью YOLO.

    Используется в эндпоинте сегментации, где можно передать изображение напрямую
    в виде base64-строки или путь к файлу на сервере.

    Attributes:
        image (Optional[str]): Изображение в формате base64. По умолчанию None.
        image_path (Optional[str]): Путь к изображению на файловой системе. По умолчанию None.
        confidence (Optional[float]): Порог уверенности модели для фильтрации предсказаний.
                                      Значение от 0 до 1. По умолчанию 0.75.
        output_folder (Optional[str]): Каталог для сохранения результатов (маски, аннотации).
                                       По умолчанию "/home/jetadmin/Apps/Cucumber/Output".
    """
    image: Optional[str] = None
    image_path: Optional[str] = None
    confidence: Optional[float] = 0.75
    output_folder: Optional[str] = "/home/jetadmin/Apps/Cucumber/Output"


class TriggerBody(BaseModel):
    """
    Модель запроса для ручного запуска процесса захвата и анализа кадра с камеры.

    Используется в эндпоинте, который инициирует однократный снимок и обработку.

    Attributes:
        classes (Optional[List[str]]): Список классов для обнаружения. По умолчанию ["Cucumber"].
        output_folder (Optional[str]): Каталог для сохранения выходных данных (изображения, маски).
                                       По умолчанию "/home/jetadmin/Apps/Cucumber/Output".
        brighten (Optional[int]): Значение яркости для постобработки изображения (условное).
                                  Может использоваться для улучшения визуализации. По умолчанию 0.
        confidence (Optional[float]): Порог уверенности модели. Значение от 0 до 1.
                                      По умолчанию 0.5.
    """
    classes: Optional[List[str]] = ["Cucumber"]
    output_folder: Optional[str] = "/home/jetadmin/Apps/Cucumber/Output"
    brighten: Optional[int] = 0
    confidence: Optional[float] = 0.5


class LoopBody(BaseModel):
    """
    Модель запроса для запуска циклического режима захвата и анализа.

    Используется для включения фонового цикла, который периодически делает снимки и обрабатывает их.

    Attributes:
        output_folder (Optional[str]): Каталог для сохранения результатов циклической обработки.
                                       По умолчанию "/home/jetadmin/Apps/Cucumber/Output".
        interval (Optional[int]): Интервал между снимками в секундах. По умолчанию 3.
    """
    output_folder: Optional[str] = "/home/jetadmin/Apps/Cucumber/Output"
    interval: Optional[int] = 3
