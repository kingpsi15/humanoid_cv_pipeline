import cv2
import numpy as np
from obstacle_detector import ObstacleDetector

def create_synthetic_track():
    # Scale: 1 cm = 5 pixels
    # Track is 1.40m wide (700px), 2.50m long (1250px)
    img = np.ones((1250, 700, 3), dtype=np.uint8) * 255
    
    # Left and Right Boundaries (10cm = 50px)
    cv2.rectangle(img, (0, 0), (50, 1250), (180, 180, 180), -1)
    cv2.rectangle(img, (650, 0), (700, 1250), (180, 180, 180), -1)
    
    # Start and Finish Lines (black, 2cm = 10px)
    cv2.rectangle(img, (50, 1240), (650, 1250), (0, 0, 0), -1) # Start
    cv2.rectangle(img, (50, 0), (650, 10), (0, 0, 0), -1) # Finish
    
    # Obstacle 1: 60cm cube (300x300px), left side, 35cm (175px) from start
    cv2.rectangle(img, (50, 765), (350, 1065), (100, 100, 200), -1)
    
    # Obstacle 2: 60cm cube, right side, 60cm (300px) gap from Obstacle 1
    cv2.rectangle(img, (350, 165), (650, 465), (100, 100, 200), -1)
    
    return img

def apply_perspective(img):
    """
    Simulates a robot's camera view by applying a perspective transform 
    to the top-down image.
    """
    h, w = img.shape[:2]
    
    # Source points (a rectangle in the lower-mid part of the track)
    # We simulate looking from near the start line
    src = np.float32([
        [0, 500],       # Top-left
        [w, 500],       # Top-right
        [0, 1200],      # Bottom-left
        [w, 1200]       # Bottom-right
    ])
    
    # Destination points (trapezoid to simulate perspective)
    dst = np.float32([
        [150, 0],       # Top-left squeezed in
        [w-150, 0],     # Top-right squeezed in
        [0, h],         # Bottom-left normal
        [w, h]          # Bottom-right normal
    ])
    
    matrix = cv2.getPerspectiveTransform(src, dst)
    perspective_img = cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    return perspective_img

def test_pipeline():
    track_topdown = create_synthetic_track()
    cv2.imwrite("synthetic_track_topdown.png", track_topdown)
    
    track_perspective = apply_perspective(track_topdown)
    cv2.imwrite("synthetic_track_perspective.png", track_perspective)
    
    detector = ObstacleDetector(
        black_line_thresh=80, 
        floor_brightness_thresh=200 # Track floor is pure white
    )
    
    print("Processing perspective frame...")
    annotated_img, result = detector.process_frame(track_perspective)
    
    print(f"Start Line Detected: {result.start_line_detected}")
    print(f"Obstacles Found: {len(result.obstacles_contours)}")
    print(f"Heading Vector: ({result.heading_x}, {result.heading_y})")
    
    cv2.imwrite("synthetic_track_annotated.png", annotated_img)
    print("Saved annotated image to synthetic_track_annotated.png")

if __name__ == "__main__":
    test_pipeline()
