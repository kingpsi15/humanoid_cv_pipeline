import cv2
import numpy as np
from weightlifting_detector import WeightliftingDetector

def create_synthetic_weightlifting_scene():
    # Image 800x600 (W x H)
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Draw floor lines (perspective - lower in image means closer to robot)
    # Pickup line (close to robot)
    cv2.rectangle(img, (100, 480), (700, 520), (0, 0, 0), -1)
    
    # Lift line (further away)
    cv2.rectangle(img, (150, 300), (650, 320), (0, 0, 0), -1)
    
    # Draw stands (light brown)
    # Left stand
    cv2.rectangle(img, (200, 150), (280, 480), (150, 200, 220), -1)
    # Right stand
    cv2.rectangle(img, (520, 150), (600, 480), (150, 200, 220), -1)
    
    # Draw CDs (dark gray vertical rectangles since we see them edge-on)
    # Left CDs
    cv2.rectangle(img, (255, 90), (285, 210), (50, 50, 50), -1)
    # Right CDs
    cv2.rectangle(img, (515, 90), (545, 210), (50, 50, 50), -1)
    
    # Draw Bar (metal rod) spanning between the CDs
    cv2.rectangle(img, (285, 140), (515, 160), (100, 100, 100), -1)
    
    return img

def test_pipeline():
    test_image = create_synthetic_weightlifting_scene()
    cv2.imwrite("synthetic_weightlifting.png", test_image)
    
    detector = WeightliftingDetector(black_line_thresh=100)
    
    print("Processing frame...")
    annotated_img, result = detector.process_frame(test_image)
    
    print(f"Lines Detected: {result.lines_detected}")
    if result.closest_line_y:
        print(f"Closest Line Y: {result.closest_line_y}")
        
    print(f"Bar Detected: {result.bar_detected}")
    if result.bar_detected:
        print(f"Left Grasp: ({result.left_grasp_x}, {result.grasp_y})")
        print(f"Right Grasp: ({result.right_grasp_x}, {result.grasp_y})")
        
    cv2.imwrite("synthetic_weightlifting_annotated.png", annotated_img)
    print("Saved annotated image to synthetic_weightlifting_annotated.png")

if __name__ == "__main__":
    test_pipeline()
