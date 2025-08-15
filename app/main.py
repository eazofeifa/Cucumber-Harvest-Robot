import sys
sys.path.append("/home/jetadmin/Apps/Cucumber/Capture/pyorbbecsdk/install/lib/")

from fastapi import FastAPI
from contextlib import asynccontextmanager
import torch
from ultralytics import YOLO
from pyorbbecsdk import *
from app.routers import inference, capture, locate, loop, configurate
from app.routers.utils import CFG_PATH, get_yaml_entry


YOLO_CHECKPOINT_PATH = f"/home/jetadmin/Apps/Cucumber/Inference/Weights/{get_yaml_entry(CFG_PATH, ['MODEL_NAME'], 'segment-cucumber-orig.pt')}"
ALIGN_MODE = 'SW'
ENABLE_SYNC = True


@asynccontextmanager
async def lifespan(app: FastAPI):

    app.state.ml_models = {}
    app.state.cam_pipelines = {}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    task = None if '.pt' in YOLO_CHECKPOINT_PATH else 'segment'
    yolo_model = YOLO(YOLO_CHECKPOINT_PATH) if not task else YOLO(YOLO_CHECKPOINT_PATH, task=task)
    try:
        yolo_model.to(device=device)
        app.state.task = None
    except:
        app.state.task = 'segment'
    app.state.ml_models["yolo"] = yolo_model
    
    config = Config()
    pipeline = Pipeline()
    device = pipeline.get_device()
    device_info = device.get_device_info()
    device_pid = device_info.get_pid()
    enable_sync = True
    color_profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = color_profile_list.get_video_stream_profile(2560, 1440, OBFormat.MJPG, 15)
    config.enable_stream(color_profile)
    depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = depth_profile_list.get_video_stream_profile(1024, 1024, OBFormat.Y16, 15) # 640, 576
    config.enable_stream(depth_profile)
    if ALIGN_MODE == 'HW':
        config.set_align_mode(OBAlignMode.HW_MODE)
    elif ALIGN_MODE == 'SW':
        config.set_align_mode(OBAlignMode.SW_MODE)
    else:
        config.set_align_mode(OBAlignMode.DISABLE)
    if ENABLE_SYNC:
        pipeline.enable_frame_sync()
    pipeline.start(config)
    app.state.cam_pipelines["pipeline"] = pipeline
    yield

    app.state.ml_models.clear()
    app.state.cam_pipelines["pipeline"].stop()
    app.state.cam_pipelines.clear()

app = FastAPI(lifespan=lifespan)

app.include_router(configurate.router)
app.include_router(inference.router)
app.include_router(capture.router)
app.include_router(locate.router)
app.include_router(loop.router)


@app.get("/healthcheck")
def check_working():
    return {"online": True, "message": "Hello World!"}

