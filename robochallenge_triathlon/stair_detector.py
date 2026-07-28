import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class StairStep:
    y_coord: int
    lines: List[Tuple[int, int, int, int]]
    
class StairDetector:
    def __init__(self, blur_ksize=(5, 5), canny_thresh1=50, canny_thresh2=150, 
                 hough_rho=1, hough_theta=np.pi/180, hough_threshold=50, 
                 hough_min_line_len=50, hough_max_line_gap=10, 
                 horizontal_angle_tolerance=10, step_grouping_tolerance=20):
        self.blur_ksize = blur_ksize
        self.canny_thresh1 = canny_thresh1
        self.canny_thresh2 = canny_thresh2
        self.hough_rho = hough_rho
        self.hough_theta = hough_theta
        self.hough_threshold = hough_threshold
        self.hough_min_line_len = hough_min_line_len
        self.hough_max_line_gap = hough_max_line_gap
        
        self.horizontal_angle_tolerance = horizontal_angle_tolerance # in degrees
        self.step_grouping_tolerance = step_grouping_tolerance # in pixels
        
    def process_frame(self, image: np.ndarray) -> Tuple[np.ndarray, List[StairStep]]:
        """
        Process the frame to detect stair steps.
        Returns the annotated image and a list of detected steps.
        """
        original_image = image.copy()
        
        # 1. Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 2. Blur
        blurred = cv2.GaussianBlur(gray, self.blur_ksize, 0)
        
        # 3. Edge Detection
        edges = cv2.Canny(blurred, self.canny_thresh1, self.canny_thresh2)
        
        # 4. Line Detection
        lines = cv2.HoughLinesP(edges, self.hough_rho, self.hough_theta, self.hough_threshold,
                                minLineLength=self.hough_min_line_len, maxLineGap=self.hough_max_line_gap)
                                
        horizontal_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.flatten()
                
                # Calculate angle in degrees
                angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                # A horizontal line has an angle near 0 or 180 degrees
                if angle <= self.horizontal_angle_tolerance or angle >= 180 - self.horizontal_angle_tolerance:
                    horizontal_lines.append((x1, y1, x2, y2))
                    
        # 5. Cluster lines by Y coordinate
        steps = self._cluster_lines_by_y(horizontal_lines)
        
        # 6. Annotate image
        annotated_image = self._annotate_image(original_image, steps)
        
        return annotated_image, steps
        
    def _cluster_lines_by_y(self, lines: List[Tuple[int, int, int, int]]) -> List[StairStep]:
        if not lines:
            return []
            
        # Sort lines by average Y coordinate (top to bottom of the image)
        lines.sort(key=lambda l: (l[1] + l[3]) / 2)
        
        steps = []
        current_step_lines = [lines[0]]
        current_y_sum = (lines[0][1] + lines[0][3]) / 2
        
        for i in range(1, len(lines)):
            line = lines[i]
            avg_y = (line[1] + line[3]) / 2
            
            # Average Y of the current cluster
            cluster_avg_y = current_y_sum / len(current_step_lines)
            
            if abs(avg_y - cluster_avg_y) <= self.step_grouping_tolerance:
                current_step_lines.append(line)
                current_y_sum += avg_y
            else:
                steps.append(StairStep(y_coord=int(cluster_avg_y), lines=current_step_lines))
                current_step_lines = [line]
                current_y_sum = avg_y
                
        # Add the last cluster
        if current_step_lines:
            cluster_avg_y = current_y_sum / len(current_step_lines)
            steps.append(StairStep(y_coord=int(cluster_avg_y), lines=current_step_lines))
            
        # Filter steps that don't have enough lines (maybe false positives)
        # Note: image coordinates have Y=0 at the top. 
        # The step nearest to the robot is at the BOTTOM of the image (highest Y).
        steps.sort(key=lambda step: step.y_coord, reverse=True)
        return steps
        
    def _annotate_image(self, image: np.ndarray, steps: List[StairStep]) -> np.ndarray:
        for i, step in enumerate(steps):
            # Nearest step gets a distinct color (Green), others Red
            color = (0, 255, 0) if i == 0 else (0, 0, 255)
            
            # Draw individual lines
            for x1, y1, x2, y2 in step.lines:
                cv2.line(image, (x1, y1), (x2, y2), color, 2)
                
            # Draw center text
            if step.lines:
                min_x = min([min(l[0], l[2]) for l in step.lines])
                cv2.putText(image, f"Step Y={step.y_coord}", (min_x, step.y_coord - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                            
        return image
