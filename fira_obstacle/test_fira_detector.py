import cv2
import numpy as np
from fira_detector import FIRADetector

def create_synthetic_fira_scene():
    # Create green field for bottom half, gray background for top half (wall/sky)
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    img[0:300, :] = (150, 150, 150) # Gray background
    img[300:600, :] = (50, 200, 50) # Green floor
    
    # Draw Yellow Boundaries (perspective lines)
    # Left boundary
    pts_left = np.array([[100, 600], [300, 300], [280, 300], [50, 600]], np.int32)
    cv2.fillPoly(img, [pts_left], (0, 255, 255))
    
    # Right boundary
    pts_right = np.array([[700, 600], [500, 300], [520, 300], [750, 600]], np.int32)
    cv2.fillPoly(img, [pts_right], (0, 255, 255))
    
    # Draw Yellow Hole (trapezoid on the floor)
    pts_hole = np.array([[350, 450], [450, 450], [480, 500], [320, 500]], np.int32)
    cv2.fillPoly(img, [pts_hole], (0, 255, 255))
    
    # Draw Blue Walls
    cv2.rectangle(img, (150, 250), (220, 400), (255, 0, 0), -1) # Left wall
    cv2.rectangle(img, (600, 350), (700, 480), (255, 0, 0), -1) # Right wall
    
    # Draw Red Gate (red board spanning between blue pillars)
    # Pillars (Blue)
    cv2.rectangle(img, (300, 200), (330, 350), (255, 0, 0), -1) # Left pillar
    cv2.rectangle(img, (470, 200), (500, 350), (255, 0, 0), -1) # Right pillar
    # Red board (spanning across)
    cv2.rectangle(img, (280, 150), (520, 200), (0, 0, 255), -1)
    
    return img

def test_segmentation():
    test_image = create_synthetic_fira_scene()
    cv2.imwrite("synthetic_fira.png", test_image)
    
    detector = FIRADetector()
    
    print("Processing frame with Structural Classification...")
    annotated_img, result = detector.process_frame(test_image)
    
    print(f"Boundaries (Yellow Lines): {len(result.boundaries)}")
    print(f"Holes (Yellow blobs): {len(result.holes)}")
    print(f"Walls (Standalone Blue): {len(result.walls)}")
    print(f"Gates (Red tops with Blue pillars): {len(result.gates)}")
    
    if result.heading_target:
        print(f"Calculated Heading Target: {result.heading_target}")
    
    cv2.imwrite("synthetic_fira_annotated_step3.png", annotated_img)
    print("Saved annotated image to synthetic_fira_annotated_step3.png")

if __name__ == "__main__":
    test_segmentation()
