from pydantic import BaseModel
from typing import List, Tuple, Optional

class SegmentBody(BaseModel):
    image: Optional[str] = None
    image_path: Optional[str] = None
    confidence: Optional[float] = 0.75
    output_folder: Optional[str] = "/home/jetadmin/Apps/Cucumber/Output"
    

class TriggerBody(BaseModel):
    classes: Optional[List[str]] = ["Cucumber"]
    output_folder: Optional[str] = "/home/jetadmin/Apps/Cucumber/Output"
    brighten: Optional[int] = 0
    confidence: Optional[float] = 0.5
    

class LoopBody(BaseModel):
    output_folder: Optional[str] = "/home/jetadmin/Apps/Cucumber/Output"
    interval: Optional[int] = 3
