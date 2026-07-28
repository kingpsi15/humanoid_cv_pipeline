import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class ObstacleResult:
    heading_x: int  # X coordinate to walk towards
    heading_y: int  # Y coordinate to walk towards
    start_line_detected: bool
    obstacles_contours: List[np.ndarray]

class ObstacleDetector:
    def __init__(self, 
                 black_line_thresh=80, 
                 floor_brightness_thresh=180,
                 depth_obstacle_thresh=500): # e.g. 500mm
        self.black_line_thresh = black_line_thresh
        self.floor_brightness_thresh = floor_brightness_thresh
        self.depth_obstacle_thresh = depth_obstacle_thresh
        
    def process_frame(self, rgb_image: np.ndarray, depth_image: Optional[np.ndarray] = None) -> Tuple[np.ndarray, ObstacleResult]:
        """
        Process the frame to detect obstacles and find a safe heading.
        If depth_image is provided, it uses it for robust obstacle detection.
        """
        annotated_image = rgb_image.copy()
        h, w = rgb_image.shape[:2]
        
        # 1. Detect Start/Stop Lines (Black lines)
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        _, black_mask = cv2.threshold(gray, self.black_line_thresh, 255, cv2.THRESH_BINARY_INV)
        
        # Find horizontal-ish lines in the black mask
        start_line_detected = False
        contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            # A line should be wide and short
            if cw > w * 0.3 and ch < 80:
                start_line_detected = True
                cv2.rectangle(annotated_image, (x, y), (x+cw, y+ch), (255, 0, 0), 2)
                cv2.putText(annotated_image, "START/STOP", (x, max(15, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 2. Detect Obstacles
        obstacle_mask = np.zeros((h, w), dtype=np.uint8)
        
        if depth_image is not None:
            # Depth based obstacle detection (closer than threshold is an obstacle)
            valid_depth = depth_image > 0
            close_objects = depth_image < self.depth_obstacle_thresh
            obstacle_mask[valid_depth & close_objects] = 255
        else:
            # RGB fallback: Assume the floor is bright white/light gray
            # Anything significantly darker, but NOT the black start line, is an obstacle
            _, bright_mask = cv2.threshold(gray, self.floor_brightness_thresh, 255, cv2.THRESH_BINARY)
            
            # Obstacles = Not Bright AND Not Black
            obstacle_mask = cv2.bitwise_not(bright_mask)
            obstacle_mask = cv2.bitwise_and(obstacle_mask, cv2.bitwise_not(black_mask))
            
            # Clean up noise
            kernel = np.ones((15, 15), np.uint8)
            obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_OPEN, kernel)
            obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_CLOSE, kernel)
            
        # Extract contours for obstacles
        obs_contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_obs_contours = []
        for cnt in obs_contours:
            if cv2.contourArea(cnt) > 1000: # Filter small noise
                valid_obs_contours.append(cnt)
                cv2.drawContours(annotated_image, [cnt], -1, (0, 0, 255), 3)
                
        # 3. Calculate Heading (Target Gap)
        # Look at the bottom half of the image where the immediate obstacles are
        scan_y = int(h * 0.75)
        scanline = obstacle_mask[scan_y, :]
        
        # Find continuous segments of 0 (free space)
        free_spaces = []
        start_idx = None
        for i, val in enumerate(scanline):
            if val == 0 and start_idx is None:
                start_idx = i
            elif val == 255 and start_idx is not None:
                free_spaces.append((start_idx, i))
                start_idx = None
        if start_idx is not None:
            free_spaces.append((start_idx, w))
            
        # Choose the largest gap
        heading_x = int(w / 2) # Default straight
        heading_y = scan_y
        
        if free_spaces:
            largest_gap = max(free_spaces, key=lambda gap: gap[1] - gap[0])
            heading_x = (largest_gap[0] + largest_gap[1]) // 2
            
            # Draw the gap line
            cv2.line(annotated_image, (largest_gap[0], scan_y), (largest_gap[1], scan_y), (0, 255, 255), 4)
            # Draw the heading arrow from bottom center
            cv2.arrowedLine(annotated_image, (int(w/2), h), (heading_x, heading_y), (0, 255, 0), 4, tipLength=0.2)
            cv2.putText(annotated_image, "HEADING", (heading_x - 40, heading_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        result = ObstacleResult(
            heading_x=heading_x,
            heading_y=heading_y,
            start_line_detected=start_line_detected,
            obstacles_contours=valid_obs_contours
        )
        
        return annotated_image, result
