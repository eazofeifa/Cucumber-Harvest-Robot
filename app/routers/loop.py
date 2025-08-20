import asyncio
import requests
import time
import os
from fastapi import APIRouter, Request
import json
from ..schemas import TriggerBody, LoopBody
from .locate import get_segment_locate_image
from .utils import CFG_PATH, get_yaml_entry, set_yaml_entry


WITH_DUMPS = get_yaml_entry(CFG_PATH, ['WITH_DUMPS', 'LOOP'], False)


router = APIRouter()


async def write_log(err, pref="/home/jetadmin/Apps/Cucumber/Errors/err_log"):
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    fname = pref + "-" + timestamp + ".txt"

    with open(fname, 'w') as f:
        f.write('Error detail - %s' % err)


async def write_results(results, output_folder):
    with open(os.path.join(output_folder, "image_coordinates_results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
        
def read_results(output_folder):
    results = None
    with open(os.path.join(output_folder, "image_coordinates_results.json")) as f:
        results = json.load(f)
    write_results({}, output_folder)
    return results


def run_loop_segment_locate(request, body, s, d, c, m):
    tic = time.perf_counter()
    results = get_segment_locate_image(request=request, body=TriggerBody(output_folder=body.output_folder))
    duration = results["duration"].get("segment", 0) + results["duration"].get("locate", 0)
    results["duration"] = {'segment_locate': duration}
    asyncio.run(write_results(results, body.output_folder))
    toc = time.perf_counter()
    secs = get_yaml_entry(CFG_PATH, ['LOOP_SECONDS'], body.interval)
    time.sleep(max(0, secs))
    s += duration
    d += toc - tic
    c += 1
    m += 1
    if m >= 1:
        print(f"{'Segment + Locate'} response in {s / c:0.4f} seconds (average), full output in {toc - tic:0.4f} seconds ({d / c:0.4f} average)")
        m = 0
    return s, d, c, m, secs


@router.post("/capture_segment_locate_loop")
def start_loop_segment_locate(request: Request, body: LoopBody):
    try:
        set_yaml_entry(CFG_PATH, ['LOOP_ACTIVE'], True)
        s, d, c, m = 0, 0, 0, 0
        go_on = True
        while go_on:
            s, d, c, m, secs = run_loop_segment_locate(request, body, s, d, c, m)
            if secs < 0:
                go_on = False
    except Exception as e:
        write_log(e)
    finally:
        set_yaml_entry(CFG_PATH, ['LOOP_ACTIVE'], False)


@router.post("/capture_segment_locate")
def get_load_segment_locate_results(request: Request, body: TriggerBody):
    secs = get_yaml_entry(CFG_PATH, ['LOOP_SECONDS'], -1)
    if secs >= 0:
        if not get_yaml_entry(CFG_PATH, ['LOOP_ACTIVE'], False):
            URL = get_yaml_entry(CFG_PATH, ['APP_HOST'], "http://[::1]:8080")
            data = {"output_folder": body.output_folder, "interval": secs}
            try:
                requests.post(f"{URL}/capture_segment_locate_loop", json=data, headers={"Content-Type": "application/json"}, timeout=secs)
            except requests.exceptions.ReadTimeout: 
                pass
        return read_results(body.output_folder)
    else:
        return get_segment_locate_image(request=request, body=body)
