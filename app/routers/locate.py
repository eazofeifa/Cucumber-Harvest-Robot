import asyncio
import time
import os
import pickle
import math
import statistics
from base64 import b64decode, b64encode
import cv2
from fastapi import APIRouter, Request
import json
import numpy as np
from typing import Union, Any, Optional
from ..schemas import TriggerBody
from .capture import get_segment_image, save_json
from .orient import get_mask_multitype, find_closest_stem
from .utils import CFG_PATH, get_yaml_entry


WITH_DUMPS = get_yaml_entry(CFG_PATH, ['WITH_DUMPS', 'LOCATE'], True)


router = APIRouter()
    

def to_pickle(obj, name, ext='.pkl'):

   with open(name + ext, 'wb') as f:

       pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def from_pickle(name, ext='.pkl'):

   res = None

   if does_exist(name + ext, ''):

       with open(name + ext, 'rb') as f:

           res = pickle.load(f)

   return res

def does_exist(name, ext='.pkl'):

   return os.path.exists(name + ext)


def arr_from_b64_str(b64_str, shape=(1440, 2560, 3)):

    tic = time.perf_counter()
    result = np.frombuffer(bytearray(b64decode(b64_str)), np.uint8).reshape(shape)
    toc = time.perf_counter()
    # print(f"b64 decoded and reshaped in {toc - tic:0.4f} seconds")

    return result


def get_first_extreme(c, oper):
    a = np.argmin(c, axis=0) if oper == 'min' else np.argmax(c, axis=0)
    return [(c[x][1], c[x][0]) for x in a]


def get_middle_extreme(c, oper):
    a = np.argwhere(c == c.min(axis=0)) if oper == 'min' else np.argwhere(c == c.max(axis=0))
    b = [[c[x].tolist() for x, y in a.tolist() if y == r] for r in [0, 1]]
    middle = lambda lst: lst[len(lst) // 2]
    center = lambda lst, k: middle(sorted(lst, key=lambda x: x[k]))
    m = [center(b[i], ii) for i, ii in enumerate([1, 0])]
    return [(x[1], x[0]) for x in m]


def get_extremes(mask, offset=(0, 0, 0, 0), auto_off=0.85):
    c = mask
    left, top = get_middle_extreme(c, 'min')
    right, bottom = get_middle_extreme(c, 'max')
    tblr = [top, bottom, left, right]
    center = (int(sum(x[0] for x in tblr) / 4), int(sum(x[1] for x in tblr) / 4))
    dist = lambda d1, d2: math.ceil(abs(d2 - d1) * (1 - auto_off))
    off = lambda w: dist(top[0], bottom[0]) if w == 'vert' else dist(left[1], right[1])
    print(tblr)
    t, b = [x if x is not None else off('vert') for x in offset[:2]]
    l, r = [x if x is not None else off('horiz') for x in offset[2:]]
    print(((top[1], t), b, l, r))
    top = (top[0] + t, top[1])
    bottom = (bottom[0] - b, bottom[1])
    left = (left[0], left[1] + l)
    right = (right[0], right[1] - r)
    return top, bottom, left, right, center, c
    
    
def get_metrics_coordinates_upper_left(
    u, v, d, 
    fx=1498.922607, fy = 1497.590820, 
    cx =1273.266724, cy=728.279663
):
    z = d
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return x, y, z


def get_metrics_coordinates(
    u, v, d, **kwargs
):
    x, y, _ = get_metrics_coordinates_upper_left(u, v, d, **kwargs)
    x0, y0, _ = get_metrics_coordinates_upper_left(1440 / 2, 2560 / 2, d, **kwargs)

    x_, y_ = y - y0, x0 - x
    h = math.sqrt(x_ ** 2 + y_ ** 2 + d ** 2)
    
    return x_, y_, d, h
    
    
def is_shape_valid(mask, active=True):
    is_valid = True
    if not active:
        return is_valid
    try:
        mask_np= mask.cpu().numpy().astype(np.uint8)
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt= max(contours, key=cv2.contourArea)
        (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
        a, b = sorted([MA, ma], reverse=True)
        if not 2.5 <= a / b <= 4.5:
            print(f'Dimensions not valid!! MA {a} ma {b} proportion {a / b}')
            is_valid = False
    except Exception as e:
        print(f'Error: {e}')
    finally:
        return is_valid
    
    
def get_coordinates(annotations, depth_data, filter_zeroes=True, filter_outliers=True):
    
    def filter_depth(d, point_list, f_zeroes, f_outliers, radius=100):
        print(f"Depth current: {d}")
        res = d
        new_p_list = [(x, y, depth_data[x - 1][y -1]) for x, y in point_list]
        trimmed_p_list = [(x, y, z) for x, y, z in new_p_list if z != 0]
        z_list = [z for x, y, z in trimmed_p_list]
        if z_list:
            if f_zeroes:
                if res == 0:
                    m = statistics.median(z_list)
                    if m > 0:
                        res = m
            if f_outliers:
                m = statistics.median(z_list)
                far, near = m + radius, m - radius
                if res > far or res < near:
                    res = m
        return res
                

    results = []
    for ix, annotation in enumerate(annotations):
        # save_json(annotation, get_yaml_entry(CFG_PATH, ['SAND_DIR']), f"annotation_{ix}.json")
        to_pickle(annotation, os.path.join(get_yaml_entry(CFG_PATH, ['SAND_DIR']), f"annotation_{ix}"))
        try:
            result = annotation

            cucumber_ixs = [i for i in range(len(result.boxes.cls)) if int(result.boxes.cls[i]) == 0]
            stem_ixs = [i for i in range(len(result.boxes.cls)) if int(result.boxes.cls[i]) == 1]
            # print(f"{len(cucumber_ixs)} num cucumbers")
            # print(f"{len(stem_ixs)} num stems")
            # print(result)
            stems_per_cucumber = None
            if result.masks:
                cucumber_mask_objs = [result.masks[x] for x in cucumber_ixs]
                print(f"{len(cucumber_mask_objs)} num cucumber mask objs")
                stem_mask_objs = [result.masks[x] for x in stem_ixs]
                print(f"{len(stem_mask_objs)} num stem mask objs")
                orig_img = cv2.cvtColor(result.orig_img, cv2.COLOR_RGB2BGR)
                cucumber_masks = [get_mask_multitype(x, orig_img, 0)[0] for x in cucumber_mask_objs]
                print(f"{len(cucumber_masks)} num cucumber masks")
                stem_masks = [get_mask_multitype(x, orig_img, 1)[0] for x in stem_mask_objs]
                print(f"{len(stem_masks)} num stem masks")
                stems_per_cucumber = [find_closest_stem(x, i, stem_masks, orig_img) for i, x in enumerate(cucumber_masks)]
                # cucumber_masks_overlaid = [get_mask_multitype(x, orig_img, 0)[2] for x in cucumber_mask_objs]
                # stem_masks_overlaid = [get_mask_multitype(x, orig_img, 1)[2] for x in stem_mask_objs]
                # vis_image, lines_image = visualize_relationships(orig_img, cucumber_masks, stem_masks, stems_per_cucumber)
            
            masks = annotation.masks.xy
            raw_masks = annotation.masks.data
        except Exception as e:
            masks, raw_masks = [], []
            print('No detection masks')
            print(e)
        for i, mask in enumerate(masks):
            if i in cucumber_ixs and is_shape_valid(raw_masks[i]):
                cucumber_stem_info = stems_per_cucumber[cucumber_ixs.index(i)] if stems_per_cucumber else None
                mask_image = mask.astype(int)
                top, bottom, left, right, center, _ = get_extremes(mask_image, (None,0,0,0))
                points = {'top': top, 'bottom': bottom, 'left': left, 'right': right, 'center': center}
                res_dict = {
                    k: (int(v[0]), int(v[1]), int(filter_depth(depth_data[v[0] - 1][v[1] -1], points.values(), filter_zeroes, filter_outliers))) for k, v in points.items()
                }
                if cucumber_stem_info:
                    res_dict.update({'side': cucumber_stem_info['side']})
                results.append(res_dict)
    if results:
        raw_results = sorted(results, key=lambda d: d['center'][2], reverse=False)
        metric_raw_results = [{k: v if k == 'side' else get_metrics_coordinates_upper_left(v[0], v[1], v[2]) for k, v in x.items()} for x in raw_results]
        metric_results = [{k: v if k == 'side' else get_metrics_coordinates(v[0], v[1], v[2]) for k, v in x.items()} for x in raw_results]
        return raw_results, metric_raw_results, metric_results
    else:
        return [], [], []


async def write_results(image_data, results, output_folder):
    with open(os.path.join(output_folder, "image_coordinates_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    cv2.imwrite(os.path.join(output_folder, "image_distance_ranked.jpg"), image_data)


def dump_results(output_folder, image_data, coordinates):
    
    tic = time.perf_counter()
    b64image = b64encode(image_data).decode('utf-8')
    toc = time.perf_counter()
    # print(f"b64 encoded in {toc - tic:0.4f} seconds")
    
    # print('')
    # print(f"Coordinates: {coordinates}")
    # print('')
    results = {
        'image': b64image, 
        'coordinates': coordinates['pixel_coordinates'],
        'all_coordinates': coordinates
    }
    if WITH_DUMPS:
        asyncio.run(write_results(image_data, results, output_folder))
    return results


def put_labels(image, coordinates):
    def get_new_image(the_image, xy, color, thickness):
        return cv2.putText(
            img = the_image,
            text = str(i),
            org = xy,
            fontFace = cv2.FONT_HERSHEY_DUPLEX,
            fontScale = 6.0,
            color = color,
            thickness = thickness
        )
    
    new_image = image
    for i, coordinate in enumerate(coordinates):
        yx = tuple(coordinate['center'][:2])
        xy = (yx[1], yx[0])
        new_image = get_new_image(new_image, xy, (255, 255, 255), 20)
        new_image = get_new_image(new_image, xy, (0, 0, 0), 5)
    return new_image


@router.post("/capture_segment_locate_default")
def get_segment_locate_image(request: Request, body: TriggerBody):
    tic = time.perf_counter()
    captures, segments, capture_segment_duration = get_segment_image(request=request, body=body)
    tic1 = time.perf_counter()
    coordinates, metric_coordinates, metric_coordinates_centered = get_coordinates(annotations=segments['annotations'], depth_data=captures['depth_data'])
    output_folder = body.output_folder
    new_image = arr_from_b64_str(segments['image'])
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    # print(f'Image is None? {new_image is None}')
    if coordinates:
        new_image = put_labels(new_image, coordinates)
    labeled_coordinates = [{"order": i, **coord} for i, coord in enumerate(coordinates)]
    labeled_metric_coordinates = [{"order": i, **coord} for i, coord in enumerate(metric_coordinates)]
    labeled_metric_coordinates_centered = [{"order": i, **coord} for i, coord in enumerate(metric_coordinates_centered)]
    all_labeled_coordinates = {
        "pixel_coordinates": coordinates,
        "metric_coordinates": metric_coordinates,
        "metric_coordinates_centered": metric_coordinates_centered
    }
    results = dump_results(output_folder, new_image, all_labeled_coordinates)
    toc = time.perf_counter()
    duration = {
        **capture_segment_duration, 
        'locate': toc - tic1,
        'capture_detect_segment_locate': toc - tic
    }
    # print(f"Received the {'Capture + Detect + Segment + Locate'} response in {toc - tic:0.4f} seconds")
    return {**results, "duration": duration}
    
    
    
