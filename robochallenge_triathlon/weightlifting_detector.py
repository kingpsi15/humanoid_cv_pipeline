import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class WeightliftingResult:
    lines_detected: int  # Number of black lines currently visible
    closest_line_y: Optional[int] # Y coordinate of the closest floor line
    bar_detected: bool
    left_grasp_x: Optional[int]
    right_grasp_x: Optional[int]
    grasp_y: Optional[int]

class WeightliftingDetector:
    def __init__(self, 
                 black_line_thresh=80):
        self.black_line_thresh = black_line_thresh
        
    def process_frame(self, image: np.ndarray) -> Tuple[np.ndarray, WeightliftingResult]:
        annotated_image = image.copy()
        h, w = image.shape[:2]
        
        # 1. Detect Floor Lines (Black lines)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, black_mask = cv2.threshold(gray, self.black_line_thresh, 255, cv2.THRESH_BINARY_INV)
        
        # Clean up noise
        kernel = np.ones((5, 5), np.uint8)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
        
        lines_detected = 0
        closest_line_y = None
        max_y = -1
        
        # Look for wide horizontal contours
        contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            # A line is wide and short
            if cw > w * 0.4 and ch < 100:
                lines_detected += 1
                cv2.rectangle(annotated_image, (x, y), (x+cw, y+ch), (255, 0, 0), 2)
                cv2.putText(annotated_image, "FLOOR LINE", (x, max(15, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                if y > max_y:
                    max_y = y
                    closest_line_y = y + ch // 2
                    
        # 2. Detect the Bar and CDs (Grasp Detection)
        # Look in the upper portion of the image assuming the bar is elevated
        roi_y_end = int(h * 0.6) # Only search top 60% of image for the bar
        roi = image[0:roi_y_end, 0:w]
        
        # Use edge detection to find the structure
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)
        edges = cv2.Canny(roi_blurred, 50, 150)
        
        # Dilate edges to connect the bar and CDs into a single solid block
        edges = cv2.dilate(edges, np.ones((9, 9), np.uint8), iterations=3)
        
        bar_detected = False
        left_grasp_x = None
        right_grasp_x = None
        grasp_y = None
        
        obj_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in obj_contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            # The bar apparatus should be quite wide, but not as wide as the floor line, and wider than it is tall
            if cw > w * 0.2 and ch < cw:
                bar_detected = True
                cv2.rectangle(annotated_image, (x, y), (x+cw, y+ch), (0, 0, 255), 2)
                cv2.putText(annotated_image, "BAR APPARATUS", (x, max(15, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # The gap is in the middle of the bar. We want to grasp inside this gap.
                padding = int(cw * 0.30) # 30% padding inwards to safely clear the CDs
                left_grasp_x = x + padding
                right_grasp_x = x + cw - padding
                grasp_y = y + ch // 2
                
                # Draw grasp targets (crosshairs)
                size = 15
                # Left target
                cv2.line(annotated_image, (left_grasp_x - size, grasp_y), (left_grasp_x + size, grasp_y), (0, 255, 0), 3)
                cv2.line(annotated_image, (left_grasp_x, grasp_y - size), (left_grasp_x, grasp_y + size), (0, 255, 0), 3)
                
                # Right target
                cv2.line(annotated_image, (right_grasp_x - size, grasp_y), (right_grasp_x + size, grasp_y), (0, 255, 0), 3)
                cv2.line(annotated_image, (right_grasp_x, grasp_y - size), (right_grasp_x, grasp_y + size), (0, 255, 0), 3)
                
                break # Just take the first valid one

        result = WeightliftingResult(
            lines_detected=lines_detected,
            closest_line_y=closest_line_y,
            bar_detected=bar_detected,
            left_grasp_x=left_grasp_x,
            right_grasp_x=right_grasp_x,
            grasp_y=grasp_y
        )
        
        return annotated_image, result
