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


WITH_DUMPS = get_yaml_entry(CFG_PATH, ['WITH_DUMPS', 'INFERENCE'], False)

router = APIRouter()


def predict(chosen_model, img, **kwargs):
    return chosen_model.predict(img, **kwargs)


def predict_and_detect(chosen_model, img, **kwargs):
    results = predict(chosen_model, img, **kwargs)

    for result in results:
        for box in result.boxes:
            cv2.rectangle(result.orig_img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)
            cv2.putText(result.orig_img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    return img, results
    

async def to_pickle(obj, name, ext='.pkl'):
    with open(name + ext, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        

async def write_image(image_data, filename):
    cv2.imwrite(filename, image_data)


def arr_from_b64_str(b64_str, shape=(1440, 2560, 3)):
    return np.frombuffer(bytearray(b64decode(b64_str)), np.uint8).reshape(shape)


def get_dump_results(image_data, image_path, width, height, annotations, output_folder):
    
    if WITH_DUMPS:
        asyncio.run(to_pickle(image_data, os.path.join(output_folder, 'image_data')))
    
    b64image = b64encode(image_data).decode('utf-8')
    
    results = {
        "image": b64image,
        "image_path": image_path,
        "annotations" : annotations,
        "box_format": "xyxy",
        "img_width": width,
        "img_height": height,
    }
    
    if WITH_DUMPS:
        asyncio.run(to_pickle(results, os.path.join(output_folder, 'annotation_segmentation_results')))
    
    return results


def get_segments_visual(results):
    background = None
    for result in results:
        height, width = result.orig_img.shape[:2]
        background = result.orig_img
        try:
            masks = result.masks.xy
        except:
            masks = []
            # print('No detection masks')
        for mask in masks:
            mask = mask.astype(int)
            cv2.drawContours(background, [mask], -1, (0, 255, 0), thickness=cv2.FILLED)
    return background


@router.post("/segment")
def segment_image(request: Request, body: SegmentBody):
    tic1 = time.perf_counter()
    
    confidence = body.confidence
    output_folder = body.output_folder
    
    if body.image is not None:
        image = arr_from_b64_str(body.image)
    else:
        image = cv2.imread(body.image_path)
        
    target_model = request.app.state.ml_models["yolo"]
    
    height, width, channels = image.shape

    params = dict(conf=confidence, imgsz=1280, save=False, show=False) # (conf=confidence, imgsz=[width, height])

    task = request.app.state.task
    print(f"The task is {task}")

    # if task:
    #     params.update({'task': task})
    
    new_image, results = predict_and_detect(target_model, image, **params)
    
    if WITH_DUMPS:
        asyncio.run(write_image(new_image, os.path.join(output_folder, "annotated_image.jpg")))
    
    toc1 = time.perf_counter()
    duration = {'detect': toc1 - tic1}
    tic2 = time.perf_counter()
    
    annotated_frame = get_segments_visual(results)
    if annotated_frame is None:
        annotated_frame = image
        print("Image annotation unsuccessful")
    annotated_output_path = os.path.join(output_folder, "annotated_segmented_image.jpg")
    if WITH_DUMPS:
        asyncio.run(write_image(annotated_frame, annotated_output_path))
    
    try:
        width, height = annotated_frame.size
    except:
        height, width, _ = annotated_frame.shape
    
    results = get_dump_results(annotated_frame, annotated_output_path, width, height, results, output_folder)
    
    toc2 = time.perf_counter()
    duration.update({
        'segment': toc2 - tic2,
        'detect_segment': toc2 - tic1
    })
    
    return results, duration
    
    

