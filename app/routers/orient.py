from sklearn.linear_model import LinearRegression
import os
import io
import cv2
from IPython.display import Image
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from scipy.spatial.distance import cdist
from tqdm import tqdm

def get_all_images(image_dir):
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend(list(Path(image_dir).glob(f'*{ext}')))
        image_files.extend(list(Path(image_dir).glob(f'*{ext.upper()}')))
    return image_files


def display_image_with_size(image_path=None, image_array=None, width=800, height=600):
    """
    Display an image with specified dimensions in a Jupyter notebook.
    
    Parameters:
    -----------
    image_path : str, optional
        Path to the image file
    image_array : numpy.ndarray, optional
        Image as numpy array (alternative to image_path)
    width : int, default=800
        Desired width of the displayed image
    height : int, default=600
        Desired height of the displayed image
    """
    if image_path is not None:
        # Load image from file
        img = PILImage.open(image_path)
        img = img.resize((width, height), PILImage.LANCZOS)
    elif image_array is not None:
        # Convert numpy array to PIL Image
        if isinstance(image_array, np.ndarray):
            img = PILImage.fromarray(
                image_array.astype('uint8') if image_array.dtype != np.uint8 else image_array
            )
            img = img.resize((width, height), PILImage.LANCZOS)
    else:
        raise ValueError("Either image_path or image_array must be provided")
    
    # Convert to bytes for display
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    
    # Display the image
    display(Image(data=buf.getvalue(), width=width, height=height))

# Example usage:
# 1. Display from file
# display_image_with_size('path/to/your/image.jpg', width=600, height=400)

# 2. Display from numpy array
# import cv2
# img_array = cv2.imread('path/to/your/image.jpg')
# img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
# display_image_with_size(image_array=img_array, width=600, height=400)


def get_mask_multitype(mask_obj, img, cls_id):

    # Colors for visualization
    class_colors = {
        0: (0, 255, 0),  # Green for cucumbers
        1: (255, 0, 0)   # Red for stems
    }
    
    assert len(mask_obj.data) == 1, "Coumpound mask found"

    assert mask_obj.orig_shape == img.shape[:2]

    mask_overlay = img.copy()

    color = class_colors[cls_id]

    img_height, img_width = mask_obj.orig_shape
    
    # Get the mask tensor and convert to numpy
    mask = mask_obj.data[0].cpu().numpy()
    
    # If mask doesn't match image dimensions, resize it
    if mask.shape[0] != img_height or mask.shape[1] != img_width:
        # Handle different mask formats:
        if len(mask.shape) == 2:
            # Binary mask - resize directly
            mask = cv2.resize(
                mask, 
                (img_width, img_height), 
                interpolation=cv2.INTER_NEAREST
            )
        else:
            # Assume it's a polygon or other format
            # Convert to binary mask first
            mask_binary = np.zeros((img_height, img_width), dtype=np.uint8)
            
            # Create mask from polygon points if available
            try:
                if hasattr(masks, 'xy') and i < len(masks.xy):
                    # Get polygon points
                    polygon_points = masks.xy[i]
                    if len(polygon_points) > 0:
                        # Convert to int32 points for fillPoly
                        pts = np.array(polygon_points, dtype=np.int32)
                        cv2.fillPoly(mask_binary, [pts], 1)
                else:
                    # Fallback to resizing the mask
                    mask_binary = cv2.resize(
                        mask.astype(np.uint8), 
                        (img_width, img_height), 
                        interpolation=cv2.INTER_NEAREST
                    )
            except Exception as e:
                print(f"Error creating mask for instance {i}: {e}")
                return
            
            mask = mask_binary

    # mask_raw = np.argwhere(mask)
    
    # Create a colored mask for overlay
    colored_mask = np.zeros_like(img)
    colored_mask[mask > 0] = color
    
    # Apply the mask to the overlay image
    alpha = 0.5  # Transparency factor
    mask_bool = mask > 0
    mask_overlay[mask_bool] = mask_overlay[mask_bool] * (1 - alpha) + colored_mask[mask_bool] * alpha

    return mask, colored_mask, mask_overlay


def determine_cucumber_side(cucumber_mask, relationship_details):
    """
    Determine the side of a cucumber in relation to a stem.
    
    Args:
        cucumber_mask: Binary mask of the cucumber
        relationship_details: Details about the relationship between cucumber and stem
    
    Returns:
        'left' or 'right'
    """
    cucumber_point = relationship_details['cucumber_point']
    stem_point = relationship_details['stem_point']
    distance_type = relationship_details['distance_type']
    stem_mask = relationship_details.get('stem_mask')
    
    # Get stem center line for orientation reference
    if stem_mask is not None:
        center_line_func, line_params = get_stem_center_line(stem_mask, None)
        m, b = line_params
    else:
        # Default to vertical orientation if no stem mask
        m, b = float('inf'), stem_point[0]
    
    # Handle based on distance type
    if distance_type == 'centroid':
        # For centroids, use the stem's center line for orientation
        if m == float('inf'):
            # Vertical line, right is positive x direction from stem
            return 'right' if cucumber_point[0] > stem_point[0] else 'left'
        else:
            # For non-vertical line, use perpendicular vector to determine side
            perp_vector = np.array([-1, m])
            
            # Ensure it points to the right of the stem when facing up the stem
            if m < 0:
                perp_vector = -perp_vector
            
            # Calculate direction from stem centroid to cucumber centroid
            direction_vector = cucumber_point - stem_point
            
            # Determine side using dot product
            dot_product = np.dot(direction_vector, perp_vector)
            return 'right' if dot_product > 0 else 'left'
    
    elif distance_type == 'mask':
        # For mask-to-mask closest points
        if m == float('inf'):
            # Vertical line, right is positive x direction
            return 'right' if cucumber_point[0] > stem_point[0] else 'left'
        else:
            # Use perpendicular to stem line to determine side
            perp_vector = np.array([-1, m])
            
            # Normalize direction
            if m < 0:
                perp_vector = -perp_vector
            
            # Direction from stem point to cucumber point
            direction_vector = cucumber_point - stem_point
            
            # Determine side
            dot_product = np.dot(direction_vector, perp_vector)
            return 'right' if dot_product > 0 else 'left'
    
    elif distance_type == 'line':
        # For line-based closest points
        if m == float('inf'):
            # Vertical line
            return 'right' if cucumber_point[0] > stem_point[0] else 'left'
        else:
            # Use the perpendicular to the stem line
            perp_vector = np.array([-1, m])
            
            # Normalize direction
            if m < 0:
                perp_vector = -perp_vector
            
            # Direction from line point to cucumber point
            direction_vector = cucumber_point - stem_point
            
            # Determine side
            dot_product = np.dot(direction_vector, perp_vector)
            return 'right' if dot_product > 0 else 'left'
    
    # Default fallback
    return None


def get_stem_center_line(stem_mask, image_shape):
    """
    Calculate a straight line that passes through the centroid of the stem mask
    and best represents its direction.
    
    Args:
        stem_mask: Binary mask of the stem
        image_shape: Shape of the original image (height, width)
    
    Returns:
        A function that takes x-coordinates and returns corresponding y-coordinates
        along the center line, as well as the parameters of the line (slope, intercept)
    """
    # Find contours of the stem mask
    contours, _ = cv2.findContours(stem_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Fallback to simple vertical line if no contours found
        center_x = stem_mask.shape[1] // 2
        return lambda x: x, (0, center_x)
    
    # Get the largest contour
    contour = max(contours, key=cv2.contourArea)
    
    # Extract points from the contour
    points = contour.reshape(-1, 2)
    
    # Calculate centroid
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = np.mean(points, axis=0).astype(int)
    
    # Fit a line to the contour points using linear regression
    if len(points) > 1:
        # Use points' y as the independent variable and x as dependent
        # This helps with vertical or near-vertical stems
        model = LinearRegression()
        X = points[:, 1].reshape(-1, 1)  # y-coordinates
        y = points[:, 0]                 # x-coordinates
        model.fit(X, y)
        
        # Get the slope and intercept for the x = f(y) line
        slope = model.coef_[0]
        intercept = model.intercept_
        
        # Convert to standard form y = mx + b for easier use
        if slope == 0:
            # Horizontal line (very rare for stems)
            m = float('inf')
            b = cy
        else:
            m = 1 / slope
            b = -intercept / slope
    else:
        # Fallback for single point
        m = 0
        b = cy
    
    # Create a function that calculates y given x using the line equation
    def center_line_func(x):
        if m == float('inf'):
            # Horizontal line, return constant y
            return np.full_like(x, b)
        else:
            return m * x + b
    
    # Return both the function and the parameters
    return center_line_func, (m, b)


def find_closest_stem(cucumber_mask, cucumber_pos, stem_masks, img, max_distance_factor=2.0, max_line_distance_factor=3.0):
    """
    Find the closest stem to a cucumber based on mask proximity.
    
    Args:
        cucumber_mask: Binary mask of the cucumber
        stems: List of stem objects with 'mask' attribute
        max_distance_factor: Maximum allowed distance as a factor of cucumber diagonal
        max_line_distance_factor: Maximum allowed distance to stem line as a factor of cucumber diagonal
        image_shape: Shape of the original image (height, width)
    
    Returns:
        Dictionary with closest stem, distance, and relationship details
    """

    image_shape = img.shape[:2]
    
    # Find cucumber contour
    cucumber_contour, _ = cv2.findContours(cucumber_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cucumber_contour:
        return None
    
    # Get cucumber bounding box and centroid
    cucumber_contour = max(cucumber_contour, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cucumber_contour)
    
    # Calculate cucumber diagonal length
    cucumber_diagonal = np.sqrt(w**2 + h**2)
    
    # Calculate cucumber centroid
    M = cv2.moments(cucumber_contour)
    if M["m00"] != 0:
        cucumber_centroid = np.array([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])])
    else:
        cucumber_centroid = np.array([x + w//2, y + h//2])
    
    # Maximum allowed distances
    max_distance = cucumber_diagonal * max_distance_factor
    max_line_distance = cucumber_diagonal * max_line_distance_factor
    
    # Extract cucumber contour points
    cucumber_points = cucumber_contour.reshape(-1, 2)
    
    closest_stem = None
    min_distance = float('inf')
    relationship_details = {}
    
    for i, stem_mask in enumerate(stem_masks):
        
        # Check if masks overlap or touch
        overlap = np.logical_and(cucumber_mask, stem_mask).any()
        
        # Find stem contour
        stem_contour, _ = cv2.findContours(stem_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not stem_contour:
            continue
        
        # Get the largest contour
        stem_contour = max(stem_contour, key=cv2.contourArea)
        
        # Calculate stem centroid
        M = cv2.moments(stem_contour)
        if M["m00"] != 0:
            stem_centroid = np.array([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])])
        else:
            s_x, s_y, s_w, s_h = cv2.boundingRect(stem_contour)
            stem_centroid = np.array([s_x + s_w//2, s_y + s_h//2])
        
        if overlap:
            # If masks overlap or touch, use centroids for distance calculation
            centroid_distance = np.linalg.norm(cucumber_centroid - stem_centroid)
            
            if centroid_distance < min_distance:
                min_distance = centroid_distance
                closest_stem = i
                relationship_details = {
                    'distance': centroid_distance,
                    'cucumber_point': cucumber_centroid,
                    'cucumber_diagonal': cucumber_diagonal,
                    'stem_point': stem_centroid,
                    'distance_type': 'centroid',
                    'stem_line': None
                }
        else:
            # If masks don't overlap, use the original logic
            stem_points = stem_contour.reshape(-1, 2)
            
            # Calculate pairwise distances between all cucumber and stem points
            distances = cdist(cucumber_points, stem_points)
            min_pair_distance = np.min(distances)
            
            # Get the indices of the closest points
            cucumber_idx, stem_idx = np.unravel_index(np.argmin(distances), distances.shape)
            closest_cucumber_point = cucumber_points[cucumber_idx]
            closest_stem_point = stem_points[stem_idx]
            
            # If distance is acceptable, consider this stem
            if min_pair_distance <= max_distance:
                if min_pair_distance < min_distance:
                    min_distance = min_pair_distance
                    closest_stem = i
                    relationship_details = {
                        'distance': min_pair_distance,
                        'cucumber_point': closest_cucumber_point,
                        'cucumber_diagonal': cucumber_diagonal,
                        'stem_point': closest_stem_point,
                        'distance_type': 'mask',
                        'stem_line': None
                    }
            else:
                # Try distance to stem center line if mask distance is too large
                center_line_func, line_params = get_stem_center_line(stem_mask, image_shape)
                
                # Calculate the distance from cucumber points to the stem center line
                m, b = line_params
                
                if m == float('inf'):
                    # Horizontal line
                    line_distances = np.abs(cucumber_points[:, 0] - b)
                else:
                    # Normal line equation: ax + by + c = 0
                    # Where a = m, b = -1, c = b
                    a, b_line, c = m, -1, b
                    # Distance formula: |ax + by + c| / sqrt(a² + b²)
                    line_distances = np.abs(a * cucumber_points[:, 0] + b_line * cucumber_points[:, 1] + c) / np.sqrt(a**2 + b_line**2)
                
                min_line_distance = np.min(line_distances)
                
                if min_line_distance <= max_line_distance and min_line_distance < min_distance:
                    min_distance = min_line_distance
                    closest_stem = i
                    
                    # Get the cucumber point closest to the line
                    closest_line_point_idx = np.argmin(line_distances)
                    closest_cucumber_point = cucumber_points[closest_line_point_idx]
                    
                    # Project cucumber point onto the stem line
                    if m == float('inf'):
                        # Horizontal line
                        closest_line_point = np.array([b, closest_cucumber_point[1]])
                    else:
                        # Calculate the closest point on the line
                        x0, y0 = closest_cucumber_point
                        x1 = (x0 + m * y0 - m * b) / (1 + m**2)
                        y1 = m * x1 + b
                        closest_line_point = np.array([x1, y1])
                    
                    relationship_details = {
                        'distance': min_line_distance,
                        'cucumber_point': closest_cucumber_point,
                        'cucumber_diagonal': cucumber_diagonal,
                        'stem_point': closest_line_point,
                        'distance_type': 'line',
                        'stem_line': line_params
                    }
    
    if closest_stem is None:
        return None
    
    return  {
        'cucumber': cucumber_pos,
        'stem': closest_stem,
        'side': determine_cucumber_side(cucumber_mask, relationship_details),
        'distance': min_distance,
        'relationship': relationship_details
    }


def visualize_relationships(image, cucumber_masks, stem_masks, relationships):
    """
    Visualize cucumber-stem relationships and sides.
    
    Args:
        image: Original image
        cucumbers: List of cucumber objects with 'mask' attribute
        stems: List of stem objects with 'mask' attribute
        relationships: List of relationship dictionaries
    
    Returns:
        List of visualization images
    """
    # Create a copy of the image for visualization
    vis_image = image.copy()
    
    # Draw masks
    for cucumber_mask in cucumber_masks:
        mask = cucumber_mask.astype(np.uint8)
        # Apply green color for cucumbers
        vis_image[mask > 0] = vis_image[mask > 0] * 0.7 + np.array([0, 255, 0]) * 0.3
    
    for stem_mask in stem_masks:
        mask = stem_mask.astype(np.uint8)
        # Apply blue color for stems
        vis_image[mask > 0] = vis_image[mask > 0] * 0.7 + np.array([255, 0, 0]) * 0.3
    
    # Create a separate image for lines
    lines_image = vis_image.copy()
    
    # Draw stem center lines
    for stem_mask in stem_masks:
        mask = stem_mask.astype(np.uint8)
        
        # Get stem center line
        center_line_func, line_params = get_stem_center_line(mask, image.shape[:2])
        m, b = line_params
        
        # Draw the center line across the entire image
        height, width = image.shape[:2]
        
        if m == float('inf'):
            # Vertical line (infinite slope)
            # b represents the x-coordinate in this case
            x_coord = int(b)
            cv2.line(lines_image, (x_coord, 0), (x_coord, height-1), (255, 255, 0), 2)
        elif abs(m) > 100:
            # Near vertical line, treat as vertical to avoid numerical issues
            # Use centroid x-coordinate
            M = cv2.moments(mask)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
            else:
                cx = width // 2
            cv2.line(lines_image, (cx, 0), (cx, height-1), (255, 255, 0), 2)
        else:
            # Normal line: calculate endpoints at image boundaries
            # Calculate y at x=0 and x=width-1
            y1 = int(m * 0 + b)
            y2 = int(m * (width-1) + b)
            
            # Ensure points are within image bounds or find intersection with image boundaries
            points = []
            
            # Check intersection with left edge (x=0)
            if 0 <= y1 < height:
                points.append((0, y1))
            
            # Check intersection with right edge (x=width-1)
            if 0 <= y2 < height:
                points.append((width-1, y2))
            
            # If we don't have enough points yet, check top and bottom edges
            if len(points) < 2:
                # Check intersection with top edge (y=0)
                if m != 0:
                    x_top = int(-b / m)
                    if 0 <= x_top < width:
                        points.append((x_top, 0))
                
                # Check intersection with bottom edge (y=height-1)
                if m != 0:
                    x_bottom = int((height-1 - b) / m)
                    if 0 <= x_bottom < width:
                        points.append((x_bottom, height-1))
            
            # Draw the line if we have at least 2 points
            if len(points) >= 2:
                cv2.line(lines_image, points[0], points[1], (255, 255, 0), 2)
    
    # Draw relationship lines and side labels
    for rel in relationships:
        if rel:
            cucumber = rel['cucumber']
            stem = rel['stem']
            relation_details = rel['relationship']
            side = rel['side']
            
            # Get the closest points
            cucumber_point = relation_details['cucumber_point'].astype(int)
            stem_point = relation_details['stem_point'].astype(int)
            
            # Draw line between closest points
            cv2.line(lines_image, tuple(cucumber_point), tuple(stem_point), (0, 255, 255), 2)
            
            # Add side label
            mid_point = ((cucumber_point[0] + stem_point[0]) // 2, 
                         (cucumber_point[1] + stem_point[1]) // 2)
            cv2.putText(lines_image, f"Cucumber: {cucumber} Stem: {stem} Side: {side}", mid_point, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    return vis_image, lines_image


def resize_image(image_array: np.ndarray, new_width: int = None, new_height: int = None) -> np.ndarray:
    """
    Resizes a NumPy image array to a specific width and height using OpenCV's cv2.resize().

    Args:
        image_array (np.ndarray): The input image as a NumPy array.
        new_width (int): The desired new width of the image.
        new_height (int): The desired new height of the image.

    Returns:
        np.ndarray: The resized image as a NumPy array.
    """
    # The dsize argument in cv2.resize takes (width, height)
    resized_image = image_array
    if new_width and new_height:
        resized_image = cv2.resize(image_array, (new_width, new_height))
    return resized_image

def load_image(file_path: str) -> np.ndarray:
    return cv2.imread(file_path)

def load_resize_image(file_path: str, new_width: int = None, new_height: int = None) -> np.ndarray:
    image_array = load_image(file_path)
    return resize_image(image_array, new_width, new_height)


def display_images_with_titles_(image1, image2, title1, title2, output_path=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    ax1.imshow(image1)
    ax1.set_title(title1, fontsize=14)
    ax1.axis('on')
    ax2.imshow(image2)
    ax2.set_title(title2, fontsize=14)
    ax2.axis('on')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    return fig

def display_images_with_titles(images, titles, output_path=None):
    fig, axs = plt.subplots(len(images), 1, figsize=(10, 12))
    for i, ax in enumerate(axs):
        ax.imshow(images[i])
        ax.set_title(titles[i], fontsize=14)
        ax.axis('on')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    return fig


def get_compare_data_images(result):
    cucumber_ixs = [i for i in range(len(result.boxes.cls)) if int(result.boxes.cls[i]) == 0]
    stem_ixs = [i for i in range(len(result.boxes.cls)) if int(result.boxes.cls[i]) == 1]
    if result.masks:
        assert all(len(x.data) == 1 for i in range(len(result.masks)) for x in result.masks[i]), "Coumpound mask(s) found"
        cucumber_mask_objs = [result.masks[x] for x in cucumber_ixs]
        stem_mask_objs = [result.masks[x] for x in stem_ixs]
        orig_img = cv2.cvtColor(result.orig_img, cv2.COLOR_RGB2BGR)
        cucumber_masks = [get_mask_multitype(x, orig_img, 0)[0] for x in cucumber_mask_objs]
        stem_masks = [get_mask_multitype(x, orig_img, 1)[0] for x in stem_mask_objs]
        stems_per_cucumber = [find_closest_stem(x, i, stem_masks, orig_img) for i, x in enumerate(cucumber_masks)]
        cucumber_masks_overlaid = [get_mask_multitype(x, orig_img, 0)[2] for x in cucumber_mask_objs]
        stem_masks_overlaid = [get_mask_multitype(x, orig_img, 1)[2] for x in stem_mask_objs]
        vis_image, lines_image = visualize_relationships(orig_img, cucumber_masks, stem_masks, stems_per_cucumber)
        return lines_image, vis_image, orig_img, stems_per_cucumber, [cucumber_masks_overlaid, stem_masks_overlaid], [cucumber_masks, stem_masks]


def get_examples_comparison(image_paths, models, titles, folder, sample_sz):
    all_compares = []
    for i in tqdm(range(sample_sz)):
        img_path = str(image_paths[i])
        image_orig = load_image(img_path)
        images_resized = [image_orig for x in titles] # load_resize_image(img_path, 1280, 1280)
        results = [m.predict(
            images_resized[j], 
            conf=0.3,
            imgsz=1280,
            save=False,
            show=False
        )[0] for j, m in enumerate(models)]
        compares = [get_compare_data_images(r) for r in results]
        all_compares.append(compares)
        lines = [None if not comp else resize_image(comp[0], 1920, 1080) for comp in compares]
        output_path = os.path.join(folder, f'example_{i}.jpg')
        final_lines = [line if line is not None else image_orig for line in lines]
        display_images_with_titles(final_lines, titles, output_path=output_path)
    return all_compares