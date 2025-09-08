from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from .utils import CFG_PATH, set_yaml_entry, get_yaml_entry

router = APIRouter()

# Константы статуса и сообщений
STATUS_OK, STATUS_FAIL = 'success', 'failure'
MSG_OK, MSG_FAIL = 'Data processed', 'Data could not be processed'


def generate_html_template(title: str, buttons: list[dict]) -> str:
    """
    Генерирует минимальный HTML-шаблон с кнопками, которые вызывают POST-запросы к FastAPI-маршрутам.

    Каждая кнопка использует JavaScript для отправки асинхронного запроса к указанному маршруту.
    При успехе показывается уведомление, при ошибке — сообщение об ошибке.

    Args:
        title (str): Заголовок страницы.
        buttons (list[dict]): Список кнопок, каждый словарь содержит:
                              - 'name' (str): отображаемое имя кнопки
                              - 'route' (str): URL-путь для POST-запроса

    Returns:
        str: Готовый HTML-код страницы.

    Example:
        buttons = [
            {"name": "Start", "route": "/start"},
            {"name": "Stop", "route": "/stop"}
        ]
    """
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .button-container {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 10px;
            margin-top: 20px;
        }}
        button {{
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s;
        }}
        button:hover {{
            background-color: #45a049;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="button-container">
"""

    for button in buttons:
        html += f"""
        <button onclick="callRoute('{button['route']}')">{button['name']}</button>"""

    html += """
    </div>

    <script>
        async function callRoute(route) {
            try {
                const response = await fetch(route, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                const result = await response.json();
                console.log('Success:', result);
                alert('Action completed successfully!');
            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred: ' + error);
            }
        }
    </script>
</body>
</html>
"""
    return html


@router.get("/", response_class=HTMLResponse)
async def get_html_page():
    """
    Возвращает HTML-страницу с интерактивными кнопками для управления конфигурацией приложения.

    Страница позволяет:
    - Управлять режимом циклического распознавания (интервалы или выключено).
    - Выбирать модель (TensorRT FP16/FP32, оригинальная).
    - Включать/отключать запись данных на диск.
    - Переключаться между реальными и тестовыми (dummy) изображениями.

    Returns:
        HTMLResponse: Готовая HTML-страница с кнопками управления.
    """
    buttons = [
        {"name": "Распознавание по запросу", "route": "/turn-off-loop"},
        {"name": "Распознавание каждые 5 сек", "route": "/loop-5-sec"},
        {"name": "Распознавание каждые 3 сек", "route": "/loop-3-sec"},
        {"name": "Распознавание каждую 1 сек", "route": "/loop-1-sec"},
        {"name": "Распознавание непрерывное", "route": "/loop-continuous"},
        {"name": "Модель TensorRT FP16", "route": "/use-tensorrt-16"},
        {"name": "Модель TensorRT FP32", "route": "/use-tensorrt-32"},
        {"name": "Модель без оптимизации", "route": "/use-original"},
        {"name": "Запись на диск", "route": "/with-dumps"},
        {"name": "Без записи на диск", "route": "/no-dumps"},
        {"name": "Использовать пример фотки", "route": "/use-dummy"},
        {"name": "Использовать реальные фотки", "route": "/use-real"}
    ]

    html_content = generate_html_template("Конфигурация", buttons)
    return html_content


def try_fail_action(meth, args) -> dict:
    """
    Выполняет функцию и возвращает статус выполнения (успех/ошибка).

    Обёртка для безопасного вызова функций, изменяющих конфигурацию.
    Используется для обработки исключений при записи в YAML.

    Args:
        meth (callable): Функция для вызова (например, set_yaml_entry).
        args (list): Аргументы, передаваемые в функцию.

    Returns:
        dict: Словарь с ключами:
              - 'status': 'success' или 'failure'
              - 'message': соответствующее сообщение
    """
    resp = {"status": STATUS_OK, "message": MSG_OK}
    try:
        meth(*args)
    except Exception as e:
        resp = {"status": STATUS_FAIL, "message": MSG_FAIL}
        print(f"Configuration update failed: {e}")
    finally:
        return resp


@router.post("/turn-off-loop")
async def turn_off_loop():
    """
    Отключает циклический режим распознавания.

    Устанавливает значение `LOOP_SECONDS` в конфигурации в -1 (выключено).

    Returns:
        dict: Статус операции.
    """
    return try_fail_action(set_yaml_entry, [CFG_PATH, ['LOOP_SECONDS'], -1])


@router.post("/loop-5-sec")
async def loop_5_sec():
    """
    Включает циклическое распознавание каждые 5 секунд.

    Устанавливает `LOOP_SECONDS = 5` в конфигурации.

    Returns:
        dict: Статус операции.
    """
    return try_fail_action(set_yaml_entry, [CFG_PATH, ['LOOP_SECONDS'], 5])


@router.post("/loop-3-sec")
async def loop_3_sec():
    """
    Включает циклическое распознавание каждые 3 секунды.

    Устанавливает `LOOP_SECONDS = 3` в конфигурации.

    Returns:
        dict: Статус операции.
    """
    return try_fail_action(set_yaml_entry, [CFG_PATH, ['LOOP_SECONDS'], 3])


@router.post("/loop-1-sec")
async def loop_1_sec():
    """
    Включает циклическое распознавание каждую секунду.

    Устанавливает `LOOP_SECONDS = 1` в конфигурации.

    Returns:
        dict: Статус операции.
    """
    return try_fail_action(set_yaml_entry, [CFG_PATH, ['LOOP_SECONDS'], 1])


@router.post("/loop-continuous")
async def loop_continuous():
    """
    Включает непрерывное (максимально частое) циклическое распознавание.

    Устанавливает `LOOP_SECONDS = 0` (режим "без задержки").

    Returns:
        dict: Статус операции.
    """
    return try_fail_action(set_yaml_entry, [CFG_PATH, ['LOOP_SECONDS'], 0])


@router.post("/use-tensorrt-16")
async def use_tensorrt_16():
    """
    Переключает модель на TensorRT с половинной точностью (FP16).

    Обновляет `MODEL_NAME` в конфигурации.

    Returns:
        dict: Статус операции.
    """
    return try_fail_action(set_yaml_entry, [CFG_PATH, ['MODEL_NAME'], 'segment-cucumber-trt-fp16.engine'])


@router.post("/use-tensorrt-32")
async def use_tensorrt_32():
    """
    Переключает модель на TensorRT с полной точностью (FP32).

    Обновляет `MODEL_NAME` в конфигурации.

    Returns:
        dict: Статус операции.
    """
    return try_fail_action(set_yaml_entry, [CFG_PATH, ['MODEL_NAME'], 'segment-cucumber-trt-fp32.engine'])


@router.post("/use-original")
async def use_original():
    """
    Переключает модель на оригинальную (неоптимизированную) версию.

    Устанавливает `MODEL_NAME = 'segment-cucumber-orig.pt'`.

    Returns:
        dict: Статус операции.
    """
    return try_fail_action(set_yaml_entry, [CFG_PATH, ['MODEL_NAME'], 'segment-cucumber-orig.pt'])


@router.post("/with-dumps")
async def with_dumps():
    """
    Включает сохранение изображений и данных на диск.

    Устанавливает `WITH_DUMPS = True`.

    Returns:
        dict: Статус операции.
    """
    return try_fail_action(set_yaml_entry, [CFG_PATH, ['WITH_DUMPS'], True])


@router.post("/no-dumps")
async def no_dumps():
    """
    Отключает сохранение изображений и данных на диск.

    Устанавливает `WITH_DUMPS = False`.

    Returns:
        dict: Статус операции.
    """
    return try_fail_action(set_yaml_entry, [CFG_PATH, ['WITH_DUMPS'], False])


@router.post("/use-dummy")
async def use_dummy():
    """
    Включает использование тестового изображения (dummy) вместо камеры.

    Полезно для отладки без подключения к камере.

    Returns:
        dict: Статус операции.
    """
    return try_fail_action(set_yaml_entry, [CFG_PATH, ['USE_DUMMY'], True])


@router.post("/use-real")
async def use_real():
    """
    Переключается на использование реальных данных с камеры (отключает dummy-режим).

    Returns:
        dict: Статус операции.
    """
    return try_fail_action(set_yaml_entry, [CFG_PATH, ['USE_DUMMY'], False])
