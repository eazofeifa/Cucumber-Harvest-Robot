import asyncio
import requests
import time
import os
import json
from fastapi import APIRouter, Request
from typing import Tuple

from ..schemas import TriggerBody, LoopBody
from .locate import get_segment_locate_image
from .utils import CFG_PATH, get_yaml_entry, set_yaml_entry


# Конфигурация: сохранять ли результаты на диск
WITH_DUMPS = get_yaml_entry(CFG_PATH, ['WITH_DUMPS', 'LOOP'], False)

router = APIRouter()


async def write_log(err: Exception, pref: str = "/home/jetadmin/Apps/Cucumber/Errors/err_log"):
    """
    Асинхронно записывает ошибку в файл с временной меткой.

    Args:
        err (Exception): Объект исключения.
        pref (str): Префикс имени файла лога.
    """
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    fname = f"{pref}-{timestamp}.txt"
    try:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, 'w') as f:
            f.write(f"Error detail - {repr(err)}\n")
    except Exception as log_err:
        print(f"Failed to write error log: {log_err}")


async def write_results(results: dict, output_folder: str):
    """
    Асинхронно сохраняет результаты в JSON-файл.

    Args:
        results (dict): Данные для сохранения.
        output_folder (str): Путь к каталогу вывода.
    """
    try:
        os.makedirs(output_folder, exist_ok=True)
        path = os.path.join(output_folder, "image_coordinates_results.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        await write_log(e)


def read_results(output_folder: str) -> dict:
    """
    Считывает результаты из JSON-файла и очищает его (записывает пустой объект).

    Args:
        output_folder (str): Путь к каталогу с результатами.

    Returns:
        dict: Результаты анализа или пустой словарь при ошибке.
    """
    path = os.path.join(output_folder, "image_coordinates_results.json")
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            # Очистка файла после чтения
            with open(path, 'w') as f:
                json.dump({}, f)
            return results
        else:
            print(f"Results file not found: {path}")
            return {}
    except Exception as e:
        print(f"Error reading results: {e}")
        return {}


def run_loop_segment_locate(
    request: Request,
    body: LoopBody,
    total_segment_locate_time: float,
    total_full_cycle_time: float,
    call_count: int,
    print_counter: int
) -> Tuple[float, float, int, int, int]:
    """
    Выполняет один цикл захвата, сегментации и локализации.

    Сохраняет результаты и контролирует интервал между итерациями.

    Args:
        request (Request): FastAPI-запрос (для передачи в другие роуты).
        body (LoopBody): Параметры цикла (папка, интервал).
        total_segment_locate_time (float): Накопленное время выполнения сегментации + локализации.
        total_full_cycle_time (float): Накопленное полное время цикла.
        call_count (int): Количество выполненных итераций.
        print_counter (int): Счётчик до вывода средних значений.

    Returns:
        tuple: Обновлённые значения (total_segment_locate_time, total_full_cycle_time, call_count, print_counter, current_interval).
    """
    tic = time.perf_counter()

    try:
        # Вызов эндпоинта сегментации и локализации
        trigger_body = TriggerBody(output_folder=body.output_folder)
        results = get_segment_locate_image(request=request, body=trigger_body)

        # Извлечение времени выполнения сегментации и локализации
        segment_time = results["duration"].get("segment", 0)
        locate_time = results["duration"].get("locate", 0)
        segment_locate_duration = segment_time + locate_time

        # Обновление общего времени
        total_segment_locate_time += segment_locate_duration

        # Сохранение результатов
        asyncio.run(write_results(results, body.output_folder))

        toc = time.perf_counter()
        full_cycle_duration = toc - tic
        total_full_cycle_time += full_cycle_duration

        # Получение текущего интервала из конфигурации
        secs = get_yaml_entry(CFG_PATH, ['LOOP_SECONDS'], body.interval)

        # Задержка между итерациями
        time.sleep(max(0, secs))

        call_count += 1
        print_counter += 1

        # Вывод средних значений каждую итерацию (можно настроить)
        if print_counter >= 1:
            avg_segment_locate = total_segment_locate_time / call_count
            avg_full_cycle = total_full_cycle_time / call_count
            print(
                f"Segment + Locate average: {avg_segment_locate:.4f}s, "
                f"Full cycle: {full_cycle_duration:.4f}s (avg {avg_full_cycle:.4f}s)"
            )
            print_counter = 0  # Сброс счётчика

        return total_segment_locate_time, total_full_cycle_time, call_count, print_counter, secs

    except Exception as e:
        asyncio.run(write_log(e))
        # В случае ошибки — продолжаем цикл, но выводим ошибку
        secs = get_yaml_entry(CFG_PATH, ['LOOP_SECONDS'], body.interval)
        time.sleep(max(0, secs))
        return total_segment_locate_time, total_full_cycle_time, call_count, print_counter, secs


@router.post("/capture_segment_locate_loop")
def start_loop_segment_locate(request: Request, body: LoopBody):
    """
    Запускает бесконечный цикл захвата и анализа огурцов.

    Работает до тех пор, пока `LOOP_SECONDS < 0`.

    Args:
        request (Request): FastAPI-запрос.
        body (LoopBody): Параметры (папка вывода, интервал по умолчанию).
    """
    try:
        set_yaml_entry(CFG_PATH, ['LOOP_ACTIVE'], True)
        total_segment_locate_time = 0.0
        total_full_cycle_time = 0.0
        call_count = 0
        print_counter = 0

        while True:
            total_segment_locate_time, total_full_cycle_time, call_count, print_counter, secs = \
                run_loop_segment_locate(request, body, total_segment_locate_time, total_full_cycle_time, call_count, print_counter)

            # Остановка цикла, если LOOP_SECONDS < 0
            if secs < 0:
                print("Loop stopped: LOOP_SECONDS < 0")
                break

    except Exception as e:
        asyncio.run(write_log(e))
    finally:
        set_yaml_entry(CFG_PATH, ['LOOP_ACTIVE'], False)
        print("Loop deactivated.")


@router.post("/capture_segment_locate")
def get_load_segment_locate_results(request: Request, body: TriggerBody):
    """
    Основной эндпоинт для получения результатов.

    Поведение зависит от режима:
    - Если `LOOP_SECONDS >= 0`: запускает или использует циклический режим.
    - Если `LOOP_SECONDS < 0`: выполняет однократную обработку.

    Args:
        request (Request): FastAPI-запрос.
        body (TriggerBody): Параметры (папка вывода).

    Returns:
        dict: Результаты анализа огурцов.
    """
    loop_seconds = get_yaml_entry(CFG_PATH, ['LOOP_SECONDS'], -1)

    if loop_seconds >= 0:
        # Циклический режим активен
        if not get_yaml_entry(CFG_PATH, ['LOOP_ACTIVE'], False):
            # Цикл не запущен — запускаем
            host = get_yaml_entry(CFG_PATH, ['APP_HOST'], "http://[::1]:8080")
            url = f"{host}/capture_segment_locate_loop"
            data = {"output_folder": body.output_folder, "interval": loop_seconds}
            try:
                # Запуск цикла (асинхронно, с таймаутом)
                requests.post(url, json=data, headers={"Content-Type": "application/json"}, timeout=loop_seconds)
            except requests.exceptions.Timeout:
                pass  # Таймаут ожидания нормален
            except Exception as e:
                asyncio.run(write_log(e))

        # Возвращаем последние результаты из файла
        return read_results(body.output_folder)

    else:
        # Однократная обработка
        return get_segment_locate_image(request=request, body=body)
