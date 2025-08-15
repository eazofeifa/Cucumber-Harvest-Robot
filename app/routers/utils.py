import yaml

import numpy as np

import cv2

from PIL import Image


CFG_PATH = '/home/jetadmin/Apps/Cucumber/Server/Loop/app/routers/config.yaml'



def resize_image_to_match(reference_image_array, target_image_path):

   """

   Load an image from file and resize it to match the dimensions of a reference image.

   

   Parameters:

   -----------

   reference_image_array : numpy.ndarray

       The reference image as a numpy array, whose dimensions will be used as the target size

   target_image_path : str

       Path to the image file that needs to be resized

       

   Returns:

   --------

   numpy.ndarray

       The resized image as a numpy array with the same dimensions as the reference image

   """

   # Get dimensions of the reference image

   if len(reference_image_array.shape) == 3:

       target_height, target_width, _ = reference_image_array.shape

   else:

       target_height, target_width = reference_image_array.shape

   

   # Method 1: Using OpenCV (faster for large images)

   try:

       # Read image using OpenCV

       img = cv2.imread(target_image_path)

       

       # Check if image was loaded successfully

       if img is None:

           raise Exception("Failed to load image with OpenCV")

           

       # If the reference image is grayscale but loaded image is color

       if len(reference_image_array.shape) == 2 and len(img.shape) == 3:

           img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

       

       # If the reference image is color but loaded image is grayscale

       elif len(reference_image_array.shape) == 3 and len(img.shape) == 2:

           img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

       

       # Resize the image to match the reference dimensions

       resized_img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

       

       # Convert from BGR to RGB if the image is color (OpenCV loads as BGR)

       if len(resized_img.shape) == 3:

           resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

           

       return resized_img

   

   except Exception as e:

       print(f"OpenCV method failed: {e}. Trying with PIL...")

       

       # Method 2: Using PIL/Pillow as a fallback

       try:

           # Load the image using PIL

           img = Image.open(target_image_path)

           

           # Resize image to match the reference dimensions

           resized_img = img.resize((target_width, target_height), Image.LANCZOS)

           

           # Convert to numpy array

           resized_array = np.array(resized_img)

           

           # Ensure the same number of channels

           if len(reference_image_array.shape) == 3 and len(resized_array.shape) == 2:

               # Convert grayscale to RGB if needed

               resized_array = np.stack((resized_array,) * 3, axis=-1)

           elif len(reference_image_array.shape) == 2 and len(resized_array.shape) == 3:

               # Convert RGB to grayscale if needed

               if resized_array.shape[2] == 3 or resized_array.shape[2] == 4:

                   resized_array = np.mean(resized_array[:, :, :3], axis=2).astype(resized_array.dtype)

           

           return resized_array

           

       except Exception as e:

           raise Exception(f"Both resize methods failed: {e}")




def read_yaml(filepath):
    """Reads data from a YAML file."""
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)
    return data if data else {}

def write_yaml(filepath, data):
    """Writes data to a YAML file."""
    with open(filepath, 'w') as file:
        yaml.dump(data, file, default_flow_style=False, sort_keys=False)

def set_yaml_entry(filepath, key_path, value):
    """
    Sets a specific entry in a YAML file.
    key_path should be a list of keys for nested structures.
    Example: ['server', 'port'] for data['server']['port']
    """
    data = read_yaml(filepath)

    current_level = data
    for i, key in enumerate(key_path):
        if i == len(key_path) - 1:
            current_level[key] = value
        else:
            if key not in current_level or not isinstance(current_level[key], dict):
                current_level[key] = {}
            current_level = current_level[key]
    write_yaml(filepath, data)

def get_yaml_entry(filepath, key_path, default=None):
    """
    Gets a specific entry in from YAML file.
    key_path should be a list of keys for nested structures.
    Example: ['server', 'port'] for data['server']['port']
    """
    data = read_yaml(filepath)

    current_level = data
    for i, key in enumerate(key_path):
        if i == len(key_path) - 1:
            return current_level.get(key, default)
        else:
            if key not in current_level or not isinstance(current_level[key], dict):
                current_level[key] = {}
            current_level = current_level[key]
